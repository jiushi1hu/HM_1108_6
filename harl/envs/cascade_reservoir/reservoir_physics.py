"""Water balance, chain feasibility and a non-anticipative terminal schedule.

The terminal-storage objective is enforced with a rolling release window built
only from information available at the decision time: current storage, today's
realised inflow, the remaining number of decisions, the terminal target and the
supplied next-seven-day mean inflow forecast. No future observed inflow is read.
"""
import numpy as np
from harl.envs.cascade_reservoir.data_loader import HEAD_MODELS


class InfeasibleCascade(RuntimeError):
    pass


TERMINAL_TOLERANCE = 1e-8


def terminal_inputs(terminal):
    """Validate one decision's terminal-scheduling inputs."""
    try:
        target = np.asarray(terminal['target'], dtype=np.float64)
        raw_days = np.asarray(terminal['remaining_days'])
        forecast = np.asarray(terminal['forecast'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('terminal requires target, remaining_days and forecast') from exc
    if (target.shape != (5,) or forecast.shape != (5,) or
            not np.isfinite(target).all() or not np.isfinite(forecast).all()):
        raise ValueError('Terminal target and forecast must be finite five-vectors')
    if (forecast < 0.).any():
        raise ValueError('Terminal forecast inflow must be nonnegative')
    if (raw_days.shape != () or not np.isfinite(raw_days) or raw_days < 1 or
            raw_days != np.floor(raw_days)):
        raise ValueError('Terminal scheduling requires a positive integer remaining horizon')
    days = int(raw_days)
    return target, days, forecast


def terminal_release_window(dt, storage, inflow, target, remaining_days, forecast, capacity):
    """Today's release window that keeps the terminal target attainable.

    Reaching the target after the remaining decisions requires a total release
    volume of V + dt*inflow + dt*(days-1)*forecast - target. Only the part that
    can still be released over the remaining days may be deferred, which gives
    the window [required - (days-1)*capacity, required]. On the final day the
    window collapses to the exact release required by the water balance.
    """
    future = (int(remaining_days)-1 if np.ndim(remaining_days) == 0 else
              np.asarray(remaining_days, dtype=np.int64)-1)
    required = storage+dt*inflow+dt*future*np.maximum(0., forecast)-target
    upper = required/dt
    return upper-future*capacity, upper


class ReservoirPhysics:
    def __init__(self, data, config):
        self.data = data
        self.config = dict(config)
        if config.get("release_mapping", "quadratic") != "quadratic":
            raise ValueError("This experiment uses quadratic normalized-action mapping")
        self.dt = float(config["timestep_seconds"])
        if self.dt != 86400.0:
            raise ValueError("This model is daily")
        eta = float(config["efficiency"])
        if not 0 < eta <= 1:
            raise ValueError("Efficiency must lie in (0, 1]")
        density = float(config["water_density_kg_m3"])
        gravity = float(config["gravity_m_s2"])
        if not np.isfinite([density, gravity]).all() or density <= 0 or gravity <= 0:
            raise ValueError("Water density and gravity must be finite and positive")
        self.head_model = config.get("head_model", "fixed")
        if self.head_model not in HEAD_MODELS:
            raise ValueError("Unknown head model: " + str(self.head_model))
        self.rho_g_eta = density * gravity * eta
        self.head_data_status = "supplied_typical_head"
        if self.head_model == "fixed":
            self.coefficient = self.rho_g_eta * data.head / 1e6
            self.effective_turbine_cap = np.minimum(data.turbine_cap, data.power_cap / self.coefficient)
        else:
            if getattr(data, "head_sensitivity", None) is None:
                raise ValueError("Load the sensitivity data before selecting storage_sensitivity")
            if not np.allclose(data.head_sensitivity["efficiency"], eta, rtol=0., atol=1e-12):
                raise ValueError("Head sensitivity efficiency must match the physical configuration")
            self.head_data_status = "assumed_sensitivity_only"
        self.reference_energy = float(24 * data.power_cap.sum())

    def _forebay_levels(self, storage):
        storage = np.asarray(storage, dtype=np.float64)
        if storage.shape != (5,) or not np.isfinite(storage).all():
            raise ValueError("Generation requires a finite five-vector of storage")
        tolerance = np.maximum(1., 1e-10 * np.abs(storage))
        if ((storage < self.data.vmin - tolerance) | (storage > self.data.vmax + tolerance)).any():
            raise InfeasibleCascade("Generation storage violates operating bounds")
        # Absorb tolerated roundoff only in the elevation lookup. Water balance
        # and the physical storage vectors are never clipped or changed.
        return self.data.levels(np.clip(storage, self.data.vmin, self.data.vmax))

    def generation(self, release, storage_start=None, storage_end=None):
        """Daily mean-head generation shared by transitions and the reference.

        The sensitivity scenario has constant efficiency and a constant combined
        tailwater/loss offset, so no flow/head root solve is required. Installed
        capacity is enforced at the modeled daily mean head.
        """
        release = np.asarray(release, dtype=np.float64)
        if (release.ndim < 1 or release.shape[-1] != 5 or
                not np.isfinite(release).all() or (release < 0).any()):
            raise ValueError("Generation releases must be finite nonnegative five-vectors")
        result = {}
        if self.head_model == "fixed":
            head = self.data.head.copy()
            coefficient, cap = self.coefficient, self.effective_turbine_cap
        else:
            start_level = self._forebay_levels(storage_start)
            end_level = self._forebay_levels(storage_end)
            mean_level = (start_level + end_level) / 2.
            scenario = self.data.head_sensitivity
            head = (scenario["reference_net_head_m"] + scenario["head_slope_m_per_m"] *
                    (mean_level - scenario["assumed_reference_forebay_level_m"]))
            if not np.isfinite(head).all() or (head <= 0).any():
                raise ValueError("Calculated net head must be finite and positive")
            coefficient = self.rho_g_eta * head / 1e6
            cap = np.minimum(self.data.turbine_cap, self.data.power_cap / coefficient)
            result["forebay_level_mean_m"] = mean_level
        turbine = np.minimum(release, cap)
        power = turbine * coefficient
        result.update(net_head_m=head, effective_turbine_cap_m3s=cap.copy(),
                      turbine=turbine, spill=release-turbine, power=power, energy=24.*power)
        return result

    def _storage_bounds(self, storage_lower, storage_upper):
        """Validate END-of-day bounds without changing the global curve domain."""
        if storage_lower is None and storage_upper is None:
            return self.data.vmin, self.data.vmax
        if storage_lower is None or storage_upper is None:
            raise ValueError("Supply both daily storage_lower and storage_upper")
        lower, upper = [np.asarray(x, dtype=np.float64) for x in (storage_lower, storage_upper)]
        if (lower.shape != (5,) or upper.shape != (5,) or
                not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any()):
            raise ValueError("Daily storage bounds must be finite ordered five-vectors")
        if (lower < self.data.vmin).any() or (upper > self.data.vmax).any():
            raise ValueError("Daily storage bounds must lie within global operating bounds")
        return lower, upper

    def downstream_intervals(self, storage, external, storage_lower=None, storage_upper=None):
        storage_min, storage_max = self._storage_bounds(storage_lower, storage_upper)
        d = self.data
        lo, hi = np.zeros(5), d.release_cap.copy()
        for i in range(3, -1, -1):
            lo[i] = max(0., lo[i + 1] - external[i + 1] + (storage_min[i + 1] - storage[i + 1]) / self.dt)
            hi[i] = min(d.release_cap[i], hi[i + 1] - external[i + 1] + (storage_max[i + 1] - storage[i + 1]) / self.dt)
            if lo[i] > hi[i] + 1e-8:
                raise InfeasibleCascade("Empty downstream-compatible release interval")
        return lo, hi

    def transition(self, storage, external, actions, storage_lower=None, storage_upper=None,
                   terminal=None):
        """Apply one action using only information available at decision time.

        Bounds constrain storage after this action. Optional terminal scheduling
        inputs (target, remaining decisions and the supplied seven-day inflow
        forecast) shrink each release interval so the terminal target stays
        attainable. The final decision collapses to the exact water-balance
        release. An empty intersection executes the nearest admissible release
        and is reported; storage is never overwritten to satisfy the target.
        """
        storage, external, actions = [np.asarray(x, dtype=np.float64) for x in (storage, external, actions)]
        if any(x.shape != (5,) or not np.isfinite(x).all() for x in (storage, external, actions)):
            raise ValueError("Expected three finite five-vectors")
        if ((actions < 0) | (actions > 1)).any() or (external < 0).any():
            raise ValueError("Invalid normalized action or external flow")
        d = self.data
        storage_min, storage_max = self._storage_bounds(storage_lower, storage_upper)
        if ((storage < d.vmin - 1) | (storage > d.vmax + 1)).any():
            raise InfeasibleCascade("Starting storage violates operating bounds")
        lower, upper = self.downstream_intervals(storage, external, storage_min, storage_max)
        inflow, release, low, high = [np.empty(5) for _ in range(4)]
        controls, executed_actions = np.ones(5, dtype=np.float32), actions.copy()
        target = days = forecast = None
        window_low, window_high = np.full(5, -np.inf), np.full(5, np.inf)
        binding, infeasible = np.zeros(5, dtype=bool), np.zeros(5, dtype=bool)
        reason = np.zeros(5, dtype=np.int8)
        if terminal is not None:
            target, days, forecast = terminal_inputs(terminal)
        for i in range(5):
            inflow[i] = external[i] + (release[i - 1] if i else 0.)
            low[i] = max(lower[i], inflow[i] + (storage[i] - storage_max[i]) / self.dt)
            high[i] = min(upper[i], inflow[i] + (storage[i] - storage_min[i]) / self.dt)
            if low[i] > high[i] + 1e-8:
                raise InfeasibleCascade("Empty feasible release interval at reservoir " + str(i))
            if terminal is not None:
                window_low[i], window_high[i] = terminal_release_window(
                    self.dt, storage[i], inflow[i], target[i], days, forecast[i],
                    d.release_cap[i])
                planned_low = max(low[i], window_low[i])
                planned_high = min(high[i], window_high[i])
                if planned_low > planned_high + TERMINAL_TOLERANCE:
                    infeasible[i] = True
                    reason[i] = 1 if window_low[i] > high[i] + TERMINAL_TOLERANCE else 2
                    release[i] = float(np.clip(0.5 * (window_low[i] + window_high[i]),
                                               low[i], high[i]))
                    controls[i] = 0
                    executed_actions[i] = 0.5
                    continue
                binding[i] = (planned_low > low[i] + TERMINAL_TOLERANCE or
                              planned_high < high[i] - TERMINAL_TOLERANCE)
                low[i], high[i] = planned_low, planned_high
            if high[i] - low[i] <= 1e-8:
                controls[i] = 0
                executed_actions[i] = 0.5
                # A tolerance-sized interval inversion can put the midpoint
                # just outside [0, outlet capacity]. Bound the actual flow;
                # storage is still computed from it and checked below.
                release[i] = min(d.release_cap[i], max(0., 0.5 * (low[i] + high[i])))
            else:
                release[i] = low[i] + executed_actions[i] ** 2 * (high[i] - low[i])
        next_storage = storage + (inflow - release) * self.dt
        residual = next_storage - storage - (inflow - release) * self.dt
        tolerance = np.maximum(1., 1e-10 * np.abs(storage))
        if ((next_storage < storage_min - tolerance) | (next_storage > storage_max + tolerance)).any():
            raise InfeasibleCascade("Storage bound violation after execution")
        if (release < -1e-8).any() or (release > d.release_cap + 1e-8).any():
            raise InfeasibleCascade("Release capacity violation")
        generated = self.generation(release, storage, next_storage)
        power, energy = generated["power"], generated["energy"]
        if (np.abs(residual) > tolerance).any() or (power > d.power_cap + 1e-7).any():
            raise AssertionError("Water/energy calculation failed")
        system_residual = next_storage.sum() - storage.sum() - (external.sum() - release[-1]) * self.dt
        if abs(system_residual) > tolerance.sum():
            raise AssertionError("Cascade water conservation failed")
        result = dict(storage_start=storage.copy(), storage_end=next_storage, inflow=inflow,
                    external=external.copy(), actions=executed_actions, release=release,
                    low=low, high=high, controls=controls, residual=residual,
                    reward=float(energy.sum() / self.reference_energy),
                    system_residual=float(system_residual), **generated)
        if storage_lower is not None:
            result.update(storage_lower_m3=storage_min.copy(), storage_upper_m3=storage_max.copy())
        if terminal is not None:
            result.update(terminal_target=target.copy(), terminal_remaining_days=days,
                terminal_forecast=forecast.copy(), terminal_window_low_m3s=window_low.copy(),
                terminal_window_high_m3s=window_high.copy(), terminal_window_binding=binding.copy(),
                terminal_window_infeasible=infeasible.copy(), terminal_window_reason=reason.copy(),
                terminal=dict(target=target.copy(), remaining_days=days,
                              forecast=forecast.copy()))
        return result
