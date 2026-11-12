"""Vectorized scalar-equivalent physics for independent counterfactual rows.

The live environment still uses ``ReservoirPhysics.transition``. This kernel
adds only a leading batch dimension and never advances an environment or
changes its inputs. The rolling terminal window, forward execution, generation
formulas and numerical tolerances match the scalar implementation.
"""
import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.reservoir_physics import (
    InfeasibleCascade, TERMINAL_TOLERANCE, terminal_release_window)


def _matrix(value, name, shape=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(IDS) or not np.isfinite(value).all():
        raise ValueError(name+' must be a finite batch of five-vectors')
    if shape is not None and value.shape != shape:
        raise ValueError('All transition inputs must have the same batch shape')
    return value


def _bounds(physics, shape, storage_lower, storage_upper):
    if storage_lower is None and storage_upper is None:
        return (np.broadcast_to(physics.data.vmin, shape),
                np.broadcast_to(physics.data.vmax, shape))
    if storage_lower is None or storage_upper is None:
        raise ValueError('Supply both daily storage_lower and storage_upper')
    values = []
    for value in (storage_lower, storage_upper):
        value = np.asarray(value, dtype=np.float64)
        if value.shape not in ((len(IDS),), shape) or not np.isfinite(value).all():
            raise ValueError('Daily storage bounds must be finite five-vectors or matching batches')
        values.append(np.broadcast_to(value, shape))
    lower, upper = values
    if (lower > upper).any():
        raise ValueError('Daily storage bounds must be ordered')
    if (lower < physics.data.vmin).any() or (upper > physics.data.vmax).any():
        raise ValueError('Daily storage bounds must lie within global operating bounds')
    return lower, upper


def _forebay_levels(physics, storage):
    tolerance = np.maximum(1., 1e-10*np.abs(storage))
    data = physics.data
    if ((storage < data.vmin-tolerance) | (storage > data.vmax+tolerance)).any():
        raise InfeasibleCascade('Generation storage violates operating bounds')
    # Clip only the level lookup within accepted roundoff, never physical water.
    lookup = np.clip(storage, data.vmin, data.vmax)
    levels = np.empty_like(lookup)
    for i, name in enumerate(IDS):
        points = data.anchors[name]
        levels[:, i] = np.interp(lookup[:, i], points[:, 1], points[:, 0])
    return levels


def _terminal_inputs(terminal, shape):
    """Broadcast the scalar terminal scheduling inputs over independent rows."""
    if terminal is None:
        return None
    try:
        target = np.asarray(terminal['target'], dtype=np.float64)
        days = np.asarray(terminal['remaining_days'], dtype=np.float64)
        forecast = np.asarray(terminal['forecast'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('terminal requires target, remaining_days and forecast') from exc
    days_shape = shape[:-1]
    if target.shape not in ((len(IDS),), shape):
        raise ValueError('terminal target must be a five-vector or matching batch')
    if forecast.shape not in ((len(IDS),), shape):
        raise ValueError('terminal forecast must be a five-vector or matching batch')
    if days.shape == (1,):
        days = days[0]
    if days.ndim == 0:
        days = np.broadcast_to(days, days_shape)
    elif days.shape != days_shape:
        raise ValueError('terminal remaining_days must be scalar or match the batch')
    if (not np.isfinite(target).all() or not np.isfinite(forecast).all() or
            not np.isfinite(days).all() or (forecast < 0.).any()):
        raise ValueError('Terminal target, forecast and remaining days must be finite')
    if (days < 1).any() or not np.equal(days, np.floor(days)).all():
        raise ValueError('Terminal scheduling requires positive integer remaining days')
    return (np.broadcast_to(target, shape), np.broadcast_to(days, days_shape).astype(np.int64),
            np.broadcast_to(forecast, shape))


def batched_transition(physics, storage, external, actions,
                       storage_lower=None, storage_upper=None, terminal=None):
    """Evaluate normalized-command transitions with input shape ``(batch, 5)``.

    Optional end-state bounds have that shape or are broadcast five-vectors.
    Result vectors have shape ``(batch, 5)``; generation rewards and cascade
    residuals have shape ``(batch,)``. A failed row raises for the entire batch.
    Empty batches are allowed. As in scalar physics, the returned ``reward`` is
    unpenalized generation; the caller applies operation-reward settings. The
    optional terminal scheduling inputs are broadcast over the batch and use
    only the same-day inflow and supplied forecast.
    """
    storage = _matrix(storage, 'Storage')
    shape = storage.shape
    external = _matrix(external, 'External flow', shape)
    actions = _matrix(actions, 'Normalized actions', shape)
    if ((actions < 0.) | (actions > 1.)).any() or (external < 0.).any():
        raise ValueError('Invalid normalized action or external flow')
    data, dt = physics.data, physics.dt
    storage_min, storage_max = _bounds(physics, shape, storage_lower, storage_upper)
    if ((storage < data.vmin-1.) | (storage > data.vmax+1.)).any():
        raise InfeasibleCascade('Starting storage violates operating bounds')
    terminal_values = _terminal_inputs(terminal, shape)

    lower = np.zeros(shape, dtype=np.float64)
    upper = np.broadcast_to(data.release_cap, shape).copy()
    for i in range(len(IDS)-2, -1, -1):
        lower[:, i] = np.maximum(0., lower[:, i+1]-external[:, i+1]+
            (storage_min[:, i+1]-storage[:, i+1])/dt)
        upper[:, i] = np.minimum(data.release_cap[i], upper[:, i+1]-external[:, i+1]+
            (storage_max[:, i+1]-storage[:, i+1])/dt)
        if (lower[:, i] > upper[:, i]+1e-8).any():
            raise InfeasibleCascade('Empty downstream-compatible release interval')

    inflow, release, low, high = [np.empty(shape, dtype=np.float64) for _ in range(4)]
    controls = np.ones(shape, dtype=np.float32)
    executed_actions = actions.copy()
    window_low = np.full(shape, -np.inf, dtype=np.float64)
    window_high = np.full(shape, np.inf, dtype=np.float64)
    binding = np.zeros(shape, dtype=bool)
    infeasible = np.zeros(shape, dtype=bool)
    reason = np.zeros(shape, dtype=np.int8)
    for i in range(len(IDS)):
        inflow[:, i] = external[:, i]+(release[:, i-1] if i else 0.)
        physical_low = np.maximum(lower[:, i], inflow[:, i]+(storage[:, i]-storage_max[:, i])/dt)
        physical_high = np.minimum(upper[:, i], inflow[:, i]+(storage[:, i]-storage_min[:, i])/dt)
        low[:, i], high[:, i] = physical_low, physical_high
        if (low[:, i] > high[:, i]+1e-8).any():
            raise InfeasibleCascade('Empty feasible release interval at reservoir '+str(i))
        if terminal_values is not None:
            target, days, forecast = terminal_values
            window_low[:, i], window_high[:, i] = terminal_release_window(
                dt, storage[:, i], inflow[:, i], target[:, i], days,
                forecast[:, i], data.release_cap[i])
            planned_low = np.maximum(low[:, i], window_low[:, i])
            planned_high = np.minimum(high[:, i], window_high[:, i])
            bad = planned_low > planned_high+TERMINAL_TOLERANCE
            infeasible[:, i] = bad
            reason[:, i] = np.where(bad,
                np.where(window_low[:, i] > high[:, i]+TERMINAL_TOLERANCE, 1, 2), 0)
            binding[:, i] = ((planned_low > low[:, i]+TERMINAL_TOLERANCE) |
                             (planned_high < high[:, i]-TERMINAL_TOLERANCE)) & ~bad
            low[:, i] = np.where(bad, physical_low, planned_low)
            high[:, i] = np.where(bad, physical_high, planned_high)
            terminal_midpoint = np.clip(0.5*(window_low[:, i]+window_high[:, i]),
                                        physical_low, physical_high)
        else:
            bad = np.zeros(shape[:-1], dtype=bool)
            terminal_midpoint = None
        forced = ~bad & (high[:, i]-low[:, i] <= 1e-8)
        uncontrolled = bad | forced
        controls[uncontrolled, i] = 0.
        executed_actions[uncontrolled, i] = .5
        midpoint = np.minimum(data.release_cap[i], np.maximum(0., .5*(low[:, i]+high[:, i])))
        regular = low[:, i]+executed_actions[:, i]**2*(high[:, i]-low[:, i])
        if terminal_midpoint is None:
            release[:, i] = np.where(forced, midpoint, regular)
        else:
            release[:, i] = np.where(bad, terminal_midpoint,
                np.where(forced, midpoint, regular))
    next_storage = storage+(inflow-release)*dt
    residual = next_storage-storage-(inflow-release)*dt
    tolerance = np.maximum(1., 1e-10*np.abs(storage))
    if ((next_storage < storage_min-tolerance) | (next_storage > storage_max+tolerance)).any():
        raise InfeasibleCascade('Storage bound violation after execution')
    if (release < -1e-8).any() or (release > data.release_cap+1e-8).any():
        raise InfeasibleCascade('Release capacity violation')

    result = {}
    if physics.head_model == 'fixed':
        head = np.broadcast_to(data.head, shape).copy()
        coefficient = physics.coefficient
        cap = np.broadcast_to(physics.effective_turbine_cap, shape).copy()
    elif physics.head_model == 'storage_sensitivity':
        start_level = _forebay_levels(physics, storage)
        end_level = _forebay_levels(physics, next_storage)
        mean_level = (start_level+end_level)/2.
        scenario = data.head_sensitivity
        head = (scenario['reference_net_head_m']+scenario['head_slope_m_per_m']*
                (mean_level-scenario['assumed_reference_forebay_level_m']))
        if not np.isfinite(head).all() or (head <= 0.).any():
            raise ValueError('Calculated net head must be finite and positive')
        coefficient = physics.rho_g_eta*head/1e6
        cap = np.minimum(data.turbine_cap, data.power_cap/coefficient)
        result['forebay_level_mean_m'] = mean_level
    else:
        raise ValueError('Unknown head model: '+str(physics.head_model))
    turbine = np.minimum(release, cap)
    power = turbine*coefficient
    energy = 24.*power
    if (np.abs(residual) > tolerance).any() or (power > data.power_cap+1e-7).any():
        raise AssertionError('Water/energy calculation failed')
    system_residual = (next_storage.sum(axis=1)-storage.sum(axis=1)-
                       (external.sum(axis=1)-release[:, -1])*dt)
    if (np.abs(system_residual) > tolerance.sum(axis=1)).any():
        raise AssertionError('Cascade water conservation failed')
    result.update(storage_start=storage.copy(), storage_end=next_storage,
        external=external.copy(), inflow=inflow, actions=executed_actions,
        release=release, low=low, high=high, controls=controls, residual=residual,
        net_head_m=head, effective_turbine_cap_m3s=cap.copy(), turbine=turbine,
        spill=release-turbine, power=power, energy=energy,
        reward=energy.sum(axis=1)/physics.reference_energy, system_residual=system_residual)
    if storage_lower is not None:
        result.update(storage_lower_m3=storage_min.copy(), storage_upper_m3=storage_max.copy())
    if terminal_values is not None:
        target, days, forecast = terminal_values
        result.update(terminal_target=target.copy(), terminal_remaining_days=days.copy(),
            terminal_forecast=forecast.copy(), terminal_window_low_m3s=window_low.copy(),
            terminal_window_high_m3s=window_high.copy(), terminal_window_binding=binding.copy(),
            terminal_window_infeasible=infeasible.copy(), terminal_window_reason=reason.copy(),
            terminal=dict(target=target.copy(), remaining_days=days.copy(),
                          forecast=forecast.copy()))
    return result
