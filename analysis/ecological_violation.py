"""Three ecological-flow violation analyses from saved inference checkpoints.

The analyses reuse the frozen replay in ``harl.envs.cascade_reservoir.checkpoint``
and the same-state attainability diagnostics in
``analysis.dispatch_diagnostics.feasibility``:

1. daily violation diagnosis: every violating reservoir-day with the actual
   release, the ecological minimum and the resulting flow deficit;
2. same-day feasible actions: the feasible release interval, the original
   command, the minimum compliant command and therefore avoidability;
3. single-action replacement replay: the minimum compliant command replaces one
   avoidable action and the saved policy is replayed to compare violations,
   storage and downstream effects.

Existing checkpoints are replayed as saved; --train-if-missing trains only the
seeds that have no checkpoint. No source dataset is changed and no existing
output is overwritten. This module lives outside harl and examples so the
implementation fingerprint stored in existing checkpoints stays valid.
"""
import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from analysis.dispatch_diagnostics.feasibility import analyze_transition
from harl.envs.cascade_reservoir.checkpoint import load_checkpoint
from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.episodes import episode_environment


DAILY_CSV = 'daily_violations.csv'
REPLAY_CSV = 'replacement_replay_summary.csv'
HEATMAP_PNG = 'daily_violation_heatmap.png'
FEASIBLE_PNG = 'feasible_action_comparison.png'
REPLAY_PNG = 'replay_comparison.png'
IDENTITY_FIELDS = ('settings_sha256', 'action_bins', 'episode_horizon_days', 'training_years',
                   'evaluation_years', 'critic_reward_layout', 'policy_update')
TRAIN_SCRIPT = ROOT/'examples'/'train_cascade.py'
TRAIN_ENVIRONMENTS = 6


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _seed_pattern(seed):
    return re.compile(r'(?:^|[^0-9])seed[-_]?0*'+str(int(seed))+r'(?![0-9])', re.IGNORECASE)


def discover_checkpoints(root, seeds, policy_update=None):
    """Find the newest update file belonging to every requested seed."""
    root = Path(root)
    files = sorted(root.rglob('update_*.pt')) if root.exists() else []
    chosen, missing, notes = {}, [], []
    for seed in seeds:
        pattern = _seed_pattern(seed)
        candidates = []
        for path in files:
            match = re.search(r'update_([0-9]+)[.]pt$', path.name)
            if match is None or pattern.search(str(path)) is None:
                continue
            update = int(match.group(1))
            if policy_update is not None and update != policy_update:
                continue
            candidates.append((update, path))
        if not candidates:
            missing.append(seed)
            continue
        update, path = max(candidates, key=lambda item: (item[0], str(item[1])))
        chosen[seed] = path
        others = sorted({value for value, _ in candidates}-{update})
        if others:
            notes.append('seed {}: updates {} available; using {}'.format(
                seed, [update]+others, update))
    return chosen, missing, notes


def verify_identical(policies, seeds):
    """Every seed must share one configuration and one evaluation sequence."""
    reference = policies[seeds[0]].metadata
    for seed in seeds:
        if int(policies[seed].metadata['seed']) != int(seed):
            raise SystemExit('Checkpoint seed mismatch: requested {} but the file stores {}'.format(
                seed, policies[seed].metadata['seed']))
    for seed in seeds[1:]:
        metadata = policies[seed].metadata
        for field in IDENTITY_FIELDS:
            if metadata[field] != reference[field]:
                raise SystemExit('Checkpoint {} differs from seed {} in {}: {!r} vs {!r}'.format(
                    seed, seeds[0], field, metadata[field], reference[field]))
        for field in ('data_hashes', 'source_hashes'):
            if metadata[field] != reference[field]:
                raise SystemExit('Checkpoint {} differs from seed {} in {}'.format(
                    seed, seeds[0], field))
    return dict(seeds=list(seeds), policy_update=int(reference['policy_update']),
                action_bins=int(reference['action_bins']),
                episode_days=int(reference['episode_horizon_days']),
                evaluation_years=list(reference['evaluation_years']),
                settings_sha256=reference['settings_sha256'])


def train_checkpoints(root, seeds, episodes, log=print):
    """Save final policies for missing seeds with the repository training entry point."""
    if episodes % TRAIN_ENVIRONMENTS:
        raise SystemExit('--train-episodes must be a multiple of {}'.format(TRAIN_ENVIRONMENTS))
    directory = Path(root)/'ecological_violation_training'
    directory.mkdir(parents=True, exist_ok=True)
    episode_log = directory/'training.jsonl'
    if episode_log.exists():
        raise SystemExit('Training log already exists: '+str(episode_log))
    command = [sys.executable, str(TRAIN_SCRIPT), '--episodes', str(episodes),
               '--seeds', *[str(seed) for seed in seeds],
               '--episode-log', str(episode_log), '--save-checkpoints',
               '--skip-final-evaluation', '--eval-interval', '1000', '--no-plot']
    log('Training missing seeds {} for {} episodes ({} updates per seed).'.format(
        list(seeds), episodes, episodes//TRAIN_ENVIRONMENTS))
    log('  '+' '.join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)


class Trajectory:
    """Daily physical states and ecological diagnostics of one frozen replay."""

    def __init__(self, requirement, timestep_seconds):
        self.requirement = np.asarray(requirement, dtype=np.float64).copy()
        self.timestep_seconds = float(timestep_seconds)
        self.dates, self.release, self.storage_start, self.storage_end = [], [], [], []
        self.inflow, self.lower, self.upper, self.actions = [], [], [], []
        self.violations = []

    def append(self, transition, commands):
        self.dates.append(str(transition['date']))
        self.release.append(np.asarray(transition['release'], dtype=np.float64).copy())
        self.storage_start.append(np.asarray(transition['storage_start'], dtype=np.float64).copy())
        self.storage_end.append(np.asarray(transition['storage_end'], dtype=np.float64).copy())
        self.inflow.append(np.asarray(transition['inflow'], dtype=np.float64).copy())
        self.lower.append(np.asarray(transition['low'], dtype=np.float64).copy())
        self.upper.append(np.asarray(transition['high'], dtype=np.float64).copy())
        self.actions.append(np.asarray(commands, dtype=np.int64).copy())

    def finish(self):
        for name in ('release', 'storage_start', 'storage_end', 'inflow', 'lower', 'upper'):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        self.actions = np.asarray(self.actions, dtype=np.int64)
        return self

    @property
    def days(self):
        return len(self.dates)

    def tolerance(self):
        return 1e-10*np.maximum(self.requirement, 1.)

    def shortfall(self):
        raw = np.maximum(self.requirement[None, :]-self.release, 0.)
        return np.where(raw > self.tolerance()[None, :], raw, 0.)

    def relative_shortfall(self):
        active = self.requirement > 0.
        return np.divide(self.shortfall(), self.requirement[None, :],
                         out=np.zeros((self.days, len(IDS))), where=active[None, :])

    def violation_days(self, start=0, stop=None):
        stop = self.days if stop is None else stop
        return (self.shortfall()[start:stop] > 0.).sum(axis=0)


class DayWriter:
    """Execute each replayed day in a fresh environment and diagnose violations.

    FrozenCascadePolicy emits five ordered reservoir records per day. Those
    commands are executed here solely to recover the authoritative transition,
    which is verified exactly against the replay before it becomes the next
    state. Counterfactual alternatives never enter the baseline trajectory.
    """

    def __init__(self, policy, source_years, trajectory, diagnose=True):
        self.policy, self.trajectory, self.diagnose = policy, trajectory, diagnose
        self.env = episode_environment(policy.data, policy.physical_config, list(source_years),
                                       policy.episode_days, policy.action_bins)
        self.pending, self.completed = [], 0
        requirements = getattr(getattr(self.env, 'operation_reward', None), 'requirements', None)
        if requirements is None:
            raise ValueError('Ecological diagnostics require loaded service requirements')
        self.requirement = np.asarray(requirements.ecological_flow_m3s, dtype=np.float64)
        self.timestep_seconds = float(self.env.physics.dt)

    def write(self, record):
        if record['reservoir'] != IDS[len(self.pending)]:
            raise ValueError('Replay records must hold five reservoirs in hydraulic order')
        if int(record['episode_step']) != self.completed+1:
            raise ValueError('Replay records must contain consecutive daily steps')
        self.pending.append(record)
        if len(self.pending) == len(IDS):
            self._finish_day()

    def _finish_day(self):
        step = self.completed+1
        commands = np.asarray([row['command_action_index'] for row in self.pending], dtype=np.int64)
        self.env.step(commands)
        transition = self.env.last_transition
        for row in self.pending:
            if row['date'] != transition['date']:
                raise AssertionError('Reconstructed source date differs from replay at step '+str(step))
        pairs = (('storage_start_m3', 'storage_start'), ('storage_end_m3', 'storage_end'),
                 ('release_m3s', 'release'), ('power_MW', 'power'), ('inflow_m3s', 'inflow'),
                 ('net_head_m', 'net_head_m'), ('level_start_m', 'levels_start'),
                 ('level_end_m', 'levels_end'))
        for field, key in pairs:
            replayed = np.asarray([row[field] for row in self.pending], dtype=np.float64)
            if not np.array_equal(replayed, np.asarray(transition[key], dtype=np.float64)):
                raise AssertionError('Baseline reconstruction differs from replay: {} at {}'.format(
                    field, transition['date']))
        if not np.array_equal(np.asarray(transition['sampled_actions'], dtype=np.int64), commands):
            raise AssertionError('Reconstructed commands differ from replay at '+str(transition['date']))
        self.trajectory.append(transition, commands)
        if self.diagnose:
            raw = np.maximum(self.requirement-np.asarray(transition['release']), 0.)
            if (raw > 1e-10*np.maximum(self.requirement, 1.)).any():
                self._diagnose(transition)
        self.pending = []
        self.completed += 1

    def _diagnose(self, transition):
        for index, row in enumerate(analyze_transition(self.env, transition)):
            if not row['ecological_flow_violating']:
                continue
            minimum = row['ecological_flow_minimum_meeting_action_index']
            self.trajectory.violations.append(dict(
                date=str(row['date']), step=self.completed+1, reservoir=str(row['reservoir_id']),
                reservoir_index=index, threshold=float(row['ecological_flow_threshold']),
                actual=float(row['ecological_flow_actual']),
                deficit=float(row['ecological_flow_shortfall']),
                deficit_volume_m3=float(row['ecological_flow_shortfall'])*self.timestep_seconds,
                relative=float(row['ecological_flow_relative_shortfall']),
                feasible_low=float(row['feasible_release_low_m3s']),
                feasible_high=float(row['feasible_release_high_m3s']),
                original_action=int(row['sampled_action_index']),
                action_bins=int(row['action_bins']),
                minimum_action=None if minimum is None else int(minimum),
                minimum_release=(None if row['ecological_flow_minimum_meeting_release_m3s'] is None
                                 else float(row['ecological_flow_minimum_meeting_release_m3s'])),
                meeting_alternatives=int(row['ecological_flow_meeting_grid_action_count'] or 0),
                minimum_achievable_deficit=float(row['ecological_flow_minimum_achievable_shortfall']),
                classification=str(row['ecological_flow_classification']),
                avoidable=str(row['ecological_flow_classification']) == 'same_day_policy_alternative'))

    def flush(self):
        if self.pending or self.completed != self.policy.episode_days:
            raise ValueError('Replay did not finish a complete horizon')


def replay_policy(policy, source_years, diagnose=True, overrides=None):
    """Replay one frozen policy and return its daily trajectory."""
    trajectory = Trajectory(np.zeros(len(IDS)), 86400.)
    writer = DayWriter(policy, source_years, trajectory, diagnose=diagnose)
    trajectory.requirement = writer.requirement.copy()
    trajectory.timestep_seconds = writer.timestep_seconds
    summary = policy.replay(source_years, writer=writer, action_overrides=overrides)
    writer.flush()
    trajectory.finish()
    if summary.get('status') != 'ok' or int(summary.get('completed_steps', 0)) != trajectory.days:
        raise AssertionError('Frozen replay did not complete the requested horizon')
    return trajectory, summary


def select_event(trajectory):
    """Most severe avoidable violation that has a downstream reservoir.

    The terminal reservoir is used only when nothing upstream is avoidable, so
    the replay comparison can always report a downstream response.
    """
    avoidable = [row for row in trajectory.violations if row['avoidable']]
    if not avoidable:
        return None
    upstream = [row for row in avoidable if row['reservoir_index'] < len(IDS)-1]
    pool = upstream or avoidable
    return min(pool, key=lambda row: (-row['deficit'], row['step'], row['reservoir_index']))


def compare_replays(seed, before, after, event):
    """Summarize one baseline trajectory against its single-command replacement."""
    index, step = event['reservoir_index'], event['step']
    replacement = event['minimum_release']
    if abs(float(after.release[step-1, index])-replacement) > 1e-6:
        raise AssertionError('Replacement replay does not reproduce the diagnosed minimum release')
    if not np.array_equal(before.release[:step-1], after.release[:step-1]):
        raise AssertionError('Replacement replay changed releases before the selected action')
    if not np.array_equal(before.actions[:step-1], after.actions[:step-1]):
        raise AssertionError('Replacement replay changed commands before the selected action')
    downstream = list(range(index+1, len(IDS)))
    residual = max(event['threshold']-replacement, 0.)
    return dict(seed=seed, date=event['date'], step=step, reservoir=event['reservoir'],
                reservoir_index=index, threshold=event['threshold'],
                original_action=event['original_action'], original_release=event['actual'],
                replacement_action=event['minimum_action'], replacement_release=replacement,
                deficit_before=event['deficit'],
                deficit_after=residual if residual > 1e-10*max(event['threshold'], 1.) else 0.,
                violation_days_before=int(before.violation_days().sum()),
                violation_days_after=int(after.violation_days().sum()),
                reservoir_days_before=int(before.violation_days()[index]),
                reservoir_days_after=int(after.violation_days()[index]),
                post_event_days_before=int(before.violation_days(step).sum()),
                post_event_days_after=int(after.violation_days(step).sum()),
                downstream=downstream,
                downstream_days_before=int(before.violation_days(step)[downstream].sum()),
                downstream_days_after=int(after.violation_days(step)[downstream].sum()),
                storage_end_before_m3=float(before.storage_end[step-1, index]),
                storage_end_after_m3=float(after.storage_end[step-1, index]),
                downstream_storage_before_m3=float(before.storage_end[step-1, downstream].sum()),
                downstream_storage_after_m3=float(after.storage_end[step-1, downstream].sum()),
                downstream_release_before_m3s=float(before.release[step-1, downstream].sum()),
                downstream_release_after_m3s=float(after.release[step-1, downstream].sum()),
                pre_event_matched=True)


def write_daily_csv(trajectories, seeds, path):
    fields = ('seed', 'date', 'episode_step', 'reservoir', 'ecological_min_m3s',
              'actual_release_m3s', 'deficit_m3s', 'deficit_volume_m3', 'relative_deficit',
              'feasible_low_m3s', 'feasible_high_m3s', 'original_action_index', 'action_bins',
              'minimum_compliant_action_index', 'minimum_compliant_release_m3s',
              'meeting_alternative_count', 'minimum_achievable_deficit_m3s',
              'classification', 'avoidable')
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator=chr(10))
        writer.writeheader()
        for seed in seeds:
            for row in trajectories[seed].violations:
                writer.writerow(dict(seed=seed, date=row['date'], episode_step=row['step'],
                    reservoir=row['reservoir'], ecological_min_m3s=row['threshold'],
                    actual_release_m3s=row['actual'], deficit_m3s=row['deficit'],
                    deficit_volume_m3=row['deficit_volume_m3'], relative_deficit=row['relative'],
                    feasible_low_m3s=row['feasible_low'], feasible_high_m3s=row['feasible_high'],
                    original_action_index=row['original_action'], action_bins=row['action_bins'],
                    minimum_compliant_action_index=row['minimum_action'],
                    minimum_compliant_release_m3s=row['minimum_release'],
                    meeting_alternative_count=row['meeting_alternatives'],
                    minimum_achievable_deficit_m3s=row['minimum_achievable_deficit'],
                    classification=row['classification'], avoidable=row['avoidable']))


def write_replay_csv(results, seeds, path):
    fields = ('seed', 'date', 'episode_step', 'reservoir', 'ecological_min_m3s',
              'original_action_index', 'original_release_m3s', 'replacement_action_index',
              'replacement_release_m3s', 'deficit_before_m3s', 'deficit_after_m3s',
              'violation_days_before', 'violation_days_after',
              'reservoir_violation_days_before', 'reservoir_violation_days_after',
              'post_event_violation_days_before', 'post_event_violation_days_after',
              'downstream_reservoirs', 'downstream_violation_days_before',
              'downstream_violation_days_after', 'affected_storage_end_before_m3',
              'affected_storage_end_after_m3', 'downstream_storage_before_m3',
              'downstream_storage_after_m3', 'downstream_release_before_m3s',
              'downstream_release_after_m3s', 'pre_event_trajectory_matched')
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator=chr(10))
        writer.writeheader()
        for seed in seeds:
            result = results.get(seed)
            if result is None:
                continue
            writer.writerow(dict(seed=seed, date=result['date'], episode_step=result['step'],
                reservoir=result['reservoir'], ecological_min_m3s=result['threshold'],
                original_action_index=result['original_action'],
                original_release_m3s=result['original_release'],
                replacement_action_index=result['replacement_action'],
                replacement_release_m3s=result['replacement_release'],
                deficit_before_m3s=result['deficit_before'],
                deficit_after_m3s=result['deficit_after'],
                violation_days_before=result['violation_days_before'],
                violation_days_after=result['violation_days_after'],
                reservoir_violation_days_before=result['reservoir_days_before'],
                reservoir_violation_days_after=result['reservoir_days_after'],
                post_event_violation_days_before=result['post_event_days_before'],
                post_event_violation_days_after=result['post_event_days_after'],
                downstream_reservoirs=';'.join(IDS[i] for i in result['downstream']),
                downstream_violation_days_before=result['downstream_days_before'],
                downstream_violation_days_after=result['downstream_days_after'],
                affected_storage_end_before_m3=result['storage_end_before_m3'],
                affected_storage_end_after_m3=result['storage_end_after_m3'],
                downstream_storage_before_m3=result['downstream_storage_before_m3'],
                downstream_storage_after_m3=result['downstream_storage_after_m3'],
                downstream_release_before_m3s=result['downstream_release_before_m3s'],
                downstream_release_after_m3s=result['downstream_release_after_m3s'],
                pre_event_trajectory_matched=result['pre_event_matched']))


def _year_spans(dates):
    spans, start, current = [], 0, dates[0][:4]
    for index, value in enumerate(dates):
        if value[:4] != current:
            spans.append((start, index, current))
            start, current = index, value[:4]
    spans.append((start, len(dates), current))
    return spans


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9., 'axes.labelsize': 8.5,
                         'savefig.dpi': 170, 'figure.constrained_layout.use': True})
    return plt


def plot_heatmap(trajectories, seeds, path):
    plt = _pyplot()
    fig, axes = plt.subplots(len(seeds), 1, figsize=(15., 2.4*len(seeds)+1.2), squeeze=False)
    image = None
    for axis, seed in zip(axes[:, 0], seeds):
        trajectory = trajectories[seed]
        image = axis.imshow(trajectory.relative_shortfall().T, aspect='auto', cmap='Reds',
                            vmin=0., vmax=1., interpolation='nearest')
        axis.set_yticks(range(len(IDS)))
        axis.set_yticklabels(IDS)
        spans = _year_spans(trajectory.dates)
        axis.set_xticks([0.5*(start+stop) for start, stop, _ in spans])
        axis.set_xticklabels([year for _, _, year in spans])
        for start, _, _ in spans[1:]:
            axis.axvline(start-0.5, color='0.35', linewidth=0.7, linestyle=':')
        counts = trajectory.violation_days()
        axis.set_title('seed {}  |  {} violating reservoir-days  ({})'.format(
            seed, int(counts.sum()),
            ', '.join('{}={}'.format(name, int(count)) for name, count in zip(IDS, counts))))
        axis.set_ylabel('Reservoir')
    axes[-1, 0].set_xlabel('Held-out evaluation sequence: one column per day, source year labelled')
    fig.colorbar(image, ax=list(axes[:, 0]), fraction=0.025, pad=0.01,
                 label='Relative deficit  (minimum - actual) / minimum')
    fig.suptitle('Daily ecological-flow violation diagnosis', fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def plot_feasible(trajectories, seeds, path):
    plt = _pyplot()
    fig, axes = plt.subplots(len(IDS), len(seeds), squeeze=False,
                             figsize=(4.4*len(seeds)+1., 2.4*len(IDS)+1.6))
    limits = []
    for index in range(len(IDS)):
        values = [trajectories[seed].requirement[index] for seed in seeds]
        for seed in seeds:
            for row in trajectories[seed].violations:
                if row['reservoir_index'] != index:
                    continue
                values.append(row['actual'])
                if row['minimum_release'] is not None:
                    values.append(row['minimum_release'])
        limits.append((0., 1.3*max(max(values), 1.)))
    for index, name in enumerate(IDS):
        for column, seed in enumerate(seeds):
            axis = axes[index, column]
            trajectory = trajectories[seed]
            events = [row for row in trajectory.violations if row['reservoir_index'] == index]
            axis.axhline(trajectory.requirement[index], color='tab:red', linewidth=0.9,
                         linestyle='--', label='ecological minimum')
            if events:
                x = np.asarray([row['step'] for row in events], dtype=np.float64)
                axis.fill_between(x, [row['feasible_low'] for row in events],
                                  [row['feasible_high'] for row in events], color='0.72',
                                  alpha=0.6, linewidth=0., label='feasible release range')
                axis.scatter(x, [row['actual'] for row in events], s=9, color='tab:orange',
                             linewidths=0., label='original release')
                compliant = [row for row in events if row['avoidable']]
                if compliant:
                    axis.scatter([row['step'] for row in compliant],
                                 [row['minimum_release'] for row in compliant], s=11,
                                 color='tab:green', linewidths=0.,
                                 label='minimum compliant release')
                axis.set_title('{} | seed {}: {} violations, {} avoidable, {} release-limited'.format(
                    name, seed, len(events), len(compliant), len(events)-len(compliant)))
            else:
                axis.text(0.5, 0.5, 'no violation in this sequence', transform=axis.transAxes,
                          ha='center', va='center', color='0.5')
                axis.set_title('{} | seed {}'.format(name, seed))
            axis.set_ylim(*limits[index])
            axis.grid(alpha=0.25, linewidth=0.6)
            if column == 0:
                axis.set_ylabel('Release  (m3/s)')
            if index == len(IDS)-1:
                axis.set_xlabel('Episode day')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Same-day feasible release range (grey, clipped at the axis limit), original '
                 'release and minimum compliant release', fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def plot_replay(results, seeds, path):
    plt = _pyplot()
    fig, axes = plt.subplots(len(seeds), 3, squeeze=False, figsize=(13.5, 3.4*len(seeds)))
    for row, seed in enumerate(seeds):
        result = results.get(seed)
        if result is None:
            for axis in axes[row]:
                axis.text(0.5, 0.5, 'no avoidable violation for seed {}'.format(seed),
                          transform=axis.transAxes, ha='center', va='center', color='0.5')
                axis.set_xticks([])
                axis.set_yticks([])
            continue
        before, after, metrics = result['before'], result['after'], result['metrics']
        vmin, vmax = result['vmin'], result['vmax']
        index, step = metrics['reservoir_index'], metrics['step']
        positions = np.arange(len(IDS))
        axes[row, 0].bar(positions-0.2, before.violation_days(), width=0.4, color='0.55',
                         label='baseline')
        axes[row, 0].bar(positions+0.2, after.violation_days(), width=0.4, color='tab:green',
                         label='replacement')
        axes[row, 0].set_xticks(positions)
        axes[row, 0].set_xticklabels(IDS)
        axes[row, 0].set_ylabel('Violation days')
        axes[row, 0].set_title('seed {}: violation days by reservoir'.format(seed))
        axes[row, 0].legend(frameon=False, fontsize=7)
        window = 120
        start, stop = max(0, step-1-window), min(before.days, step-1+window)
        days = np.arange(start+1, stop+1)
        span = float(vmax[index]-vmin[index])
        base = 100.*(before.storage_end[start:stop, index]-vmin[index])/span
        repl = 100.*(after.storage_end[start:stop, index]-vmin[index])/span
        axes[row, 1].plot(days, base, color='0.45', label='baseline')
        axes[row, 1].plot(days, repl, color='tab:green', linestyle='-', label='replacement')
        pad = 0.12*max(float(base.max()-base.min()), 0.5)
        axes[row, 1].set_ylim(max(0., float(min(base.min(), repl.min()))-pad),
                              min(100., float(max(base.max(), repl.max()))+pad))
        axes[row, 1].axvline(step, color='tab:red', linestyle=':', linewidth=1.)
        axes[row, 1].set_ylabel('Storage  (% of active capacity)')
        axes[row, 1].text(0.02, 0.06,
                          'storage change at event: {:+.1f} Mm3 ({:+.4f}% of active)'.format(
                              (metrics['storage_end_after_m3']-metrics['storage_end_before_m3'])/1e6,
                              100.*(metrics['storage_end_after_m3']-metrics['storage_end_before_m3'])
                              /max(span, 1.)),
                          transform=axes[row, 1].transAxes, fontsize=7,
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))
        axes[row, 1].set_xlabel('Episode day')
        axes[row, 1].set_title('seed {}: {} storage near the replaced action'.format(
            seed, metrics['reservoir']))
        axes[row, 1].legend(frameon=False, fontsize=7)
        days = np.arange(step, before.days+1)
        daily_before = (before.shortfall()[step-1:] > 0.).sum(axis=1)
        daily_after = (after.shortfall()[step-1:] > 0.).sum(axis=1)
        axes[row, 2].step(days, np.cumsum(daily_before), where='post', color='0.45',
                          label='baseline')
        axes[row, 2].step(days, np.cumsum(daily_after), where='post', color='tab:green',
                          label='replacement')
        axes[row, 2].set_ylabel('Cumulative violation days')
        axes[row, 2].set_xlabel('Episode day')
        axes[row, 2].set_title('seed {}: violations after the replaced action'.format(seed))
        axes[row, 2].legend(frameon=False, fontsize=7, loc='lower right')
        downstream = ';'.join(IDS[i] for i in metrics['downstream']) or 'none'
        axes[row, 2].text(0.02, 0.97,
                          'after event, all reservoirs: {:d} -> {:d} days\n'
                          'downstream {}: {:d} -> {:d} days\n'
                          'downstream storage {:.3f} -> {:.3f} bn m3'.format(
                              metrics['post_event_days_before'], metrics['post_event_days_after'],
                              downstream, metrics['downstream_days_before'],
                              metrics['downstream_days_after'],
                              metrics['downstream_storage_before_m3']/1e9,
                              metrics['downstream_storage_after_m3']/1e9),
                          transform=axes[row, 2].transAxes, va='top', fontsize=7,
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))
    fig.suptitle('Single-action replacement replay: minimum compliant command on one avoidable '
                 'violation per seed', fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def run_analysis(policies, seeds, source_years, output_dir, log=print):
    """Run the three analyses for already loaded, verified checkpoints."""
    output_dir = Path(output_dir)
    trajectories, results = {}, {}
    for seed in seeds:
        log('Replaying seed {} over source years {} ...'.format(seed, list(source_years)))
        trajectory, _ = replay_policy(policies[seed], list(source_years), diagnose=True)
        trajectories[seed] = trajectory
        event = select_event(trajectory)
        if event is None:
            log('  seed {}: no avoidable violation; replacement replay skipped'.format(seed))
            results[seed] = None
            continue
        overrides = {(event['step'], event['reservoir']): event['minimum_action']}
        replacement, _ = replay_policy(policies[seed], list(source_years), diagnose=False,
                                       overrides=overrides)
        metrics = compare_replays(seed, trajectory, replacement, event)
        results[seed] = dict(metrics=metrics, before=trajectory, after=replacement,
                             vmin=np.asarray(policies[seed].data.vmin, dtype=np.float64),
                             vmax=np.asarray(policies[seed].data.vmax, dtype=np.float64))
        log('  seed {}: {} violation days; replacing {} on step {} ({}): '
            '{:.1f} -> {:.1f} m3/s'.format(
                seed, metrics['violation_days_before'], event['reservoir'], metrics['step'],
                event['date'], metrics['original_release'], metrics['replacement_release']))
    write_daily_csv(trajectories, seeds, output_dir/DAILY_CSV)
    write_replay_csv({seed: None if results[seed] is None else results[seed]['metrics']
                      for seed in seeds}, seeds, output_dir/REPLAY_CSV)
    plot_heatmap(trajectories, seeds, output_dir/HEATMAP_PNG)
    plot_feasible(trajectories, seeds, output_dir/FEASIBLE_PNG)
    plot_replay(results, seeds, output_dir/REPLAY_PNG)
    return dict(seeds=list(seeds), source_years=list(source_years),
                violation_days={seed: int(trajectories[seed].violation_days().sum())
                                for seed in seeds},
                avoidable_days={seed: sum(1 for row in trajectories[seed].violations
                                          if row['avoidable']) for seed in seeds})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT/'runs',
                        help='Directory searched recursively for seed folders holding update files')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--policy-update', type=int, help='Exact update; default is the newest')
    parser.add_argument('--source-years', nargs='+', type=int,
                        help='Ordered source years; default is the saved evaluation sequence')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'runs'/'ecological_violation_seeds123',
                        help='New directory for two CSVs and three PNGs; never overwritten')
    parser.add_argument('--train-if-missing', action='store_true',
                        help='Train and save the missing seeds before analysing them')
    parser.add_argument('--train-episodes', type=int, default=300,
                        help='Episodes per seed when --train-if-missing is used (multiple of 6)')
    args = parser.parse_args(argv)
    seeds = list(dict.fromkeys(args.seeds))
    if not seeds:
        parser.error('at least one seed is required')
    root = Path(args.checkpoint_root).resolve()
    found, missing, notes = discover_checkpoints(root, seeds, args.policy_update)
    print('Checkpoints searched under '+str(root))
    for note in notes:
        print('  '+note)
    for seed in seeds:
        print('  seed {}: {}'.format(seed, found.get(seed, 'MISSING')))
    if missing:
        print('Required checkpoints are missing for seeds '+str(missing)+'.')
        if not args.train_if_missing:
            print('Create them with --save-checkpoints (examples/train_cascade.py or '
                  'examples/train_cascade_compare.py), or rerun this command with '
                  '--train-if-missing to train them here.')
            return 2
        train_checkpoints(root, missing, args.train_episodes)
        found, missing, notes = discover_checkpoints(root, seeds, args.policy_update)
        if missing:
            raise SystemExit('Training finished but checkpoints are still missing for '+str(missing))
        for seed in seeds:
            print('  seed {}: {}'.format(seed, found[seed]))
    policies = {seed: load_checkpoint(path) for seed, path in found.items()}
    identity = verify_identical(policies, seeds)
    years = [int(year) for year in (args.source_years or identity['evaluation_years'])]
    if len(set(years)) != len(years):
        parser.error('--source-years must be unique')
    print('Configuration: update={} action_bins={} horizon={} evaluation_years={}'.format(
        identity['policy_update'], identity['action_bins'], identity['episode_days'],
        identity['evaluation_years']))
    for seed in seeds:
        print('  seed {} checkpoint sha256 {}'.format(seed, _sha256(found[seed])))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    run_analysis(policies, seeds, years, output_dir)
    print('Outputs written to '+str(output_dir))
    for name in (DAILY_CSV, REPLAY_CSV, HEATMAP_PNG, FEASIBLE_PNG, REPLAY_PNG):
        target = output_dir/name
        print('  {}  {:.1f} kB'.format(target, target.stat().st_size/1024.))
    return 0


if __name__ == '__main__':
    sys.exit(main())
