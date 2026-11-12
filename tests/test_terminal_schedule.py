"""Terminal schedule tests."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.episodes import episode_environment


class TerminalScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.physical = yaml.safe_load(
            (ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text(encoding='utf-8'))
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical)

    def test_future_observed_inflows_do_not_affect_earlier_transitions(self):
        days, changed_from = 40, 31
        commands = ((np.arange(5)*7+13) % 31).astype(np.int64)
        original = episode_environment(self.data, self.physical, [1965], days=days,
                                       action_bins=31)
        altered_data = copy.copy(self.data)
        altered_data.external = self.data.external.copy()
        altered_data.external[changed_from:] += 10000.
        altered = episode_environment(altered_data, self.physical, [1965], days=days,
                                      action_bins=31)

        for step in range(changed_from):
            np.testing.assert_array_equal(original.storage, altered.storage)
            np.testing.assert_array_equal(original.state(), altered.state())
            np.testing.assert_array_equal(original.actor_observations(), altered.actor_observations())
            left = original.step(commands)
            right = altered.step(commands)
            np.testing.assert_array_equal(left[2], right[2])
            np.testing.assert_array_equal(left[3], right[3])
            for field in ('storage_start', 'storage_end', 'external', 'inflow', 'release',
                          'storage_lower_m3', 'storage_upper_m3', 'low', 'high', 'controls',
                          'terminal_target', 'terminal_remaining_days', 'terminal_forecast',
                          'terminal_window_low_m3s', 'terminal_window_high_m3s',
                          'terminal_window_infeasible'):
                np.testing.assert_array_equal(original.last_transition[field],
                                              altered.last_transition[field])

        original.step(commands)
        altered.step(commands)
        self.assertFalse(np.array_equal(original.last_transition['external'],
                                        altered.last_transition['external']))
        self.assertFalse(np.array_equal(original.last_transition['inflow'],
                                        altered.last_transition['inflow']))
        self.assertFalse(np.array_equal(original.last_transition['storage_end'],
                                        altered.last_transition['storage_end']))

    def test_daily_mass_balance_and_physical_hard_constraints(self):
        env = episode_environment(self.data, self.physical, [1965], days=20, action_bins=31)
        for step in range(20):
            command = (17+37*np.arange(5)+11*step) % env.action_bins
            env.step(command)
            row = env.last_transition
            tolerance = np.maximum(1., 1e-10*np.abs(row['storage_start']))
            np.testing.assert_allclose(row['storage_end']-row['storage_start'],
                (row['inflow']-row['release'])*env.physics.dt, rtol=0., atol=1e-4)
            np.testing.assert_array_equal(row['inflow'][1:],
                                          row['external'][1:]+row['release'][:-1])
            self.assertTrue(np.all(row['storage_end'] >= row['storage_lower_m3']-tolerance))
            self.assertTrue(np.all(row['storage_end'] <= row['storage_upper_m3']+tolerance))
            self.assertTrue(np.all(row['release'] >= row['low']-1e-8))
            self.assertTrue(np.all(row['release'] <= row['high']+1e-8))
            self.assertTrue(np.all(row['release'] >= -1e-8))
            self.assertTrue(np.all(row['release'] <= self.data.release_cap+1e-8))
            self.assertTrue(np.all(np.isin(row['controls'], (0., 1.))))
            self.assertTrue(np.all(row['terminal_window_infeasible'] <= (row['controls'] == 0.)))

    def test_terminal_deviation_is_reported_without_overwriting_storage(self):
        feasible = episode_environment(self.data, self.physical, [1965], days=1,
                                       action_bins=31)
        initial = feasible.storage.copy()
        feasible.step(np.full(5, 30))
        row = feasible.last_transition
        report = feasible.terminal_report
        self.assertTrue(report['met'])
        np.testing.assert_array_equal(feasible.storage, initial)
        np.testing.assert_allclose(report['deviation_m3'], np.zeros(5), rtol=0., atol=1e-6)
        np.testing.assert_allclose(row['release'], row['inflow'], rtol=0., atol=1e-7)
        self.assertFalse(np.any(row['terminal_window_infeasible']))

        physical = dict(self.physical, use_daily_level_limits=False)
        infeasible = episode_environment(self.data, physical, [1965], days=1,
                                         action_bins=31)
        infeasible.storage[0] = self.data.vmin[0]
        infeasible.step(np.full(5, 30))
        row = infeasible.last_transition
        report = infeasible.terminal_report
        expected = row['storage_end']-infeasible.initial_storage
        self.assertTrue(bool(row['terminal_window_infeasible'][0]))
        self.assertEqual(report['infeasible_days_by_reservoir'][0], 1)
        self.assertEqual(report['infeasible_steps'], 1)
        self.assertFalse(report['met'])
        self.assertFalse(report['storage_overwritten'])
        np.testing.assert_allclose(report['deviation_m3'], expected, rtol=0., atol=1e-6)
        self.assertEqual(report['max_abs_deviation_m3'], float(np.abs(expected).max()))
        np.testing.assert_allclose(row['storage_end']-row['storage_start'],
            (row['inflow']-row['release'])*infeasible.physics.dt, rtol=0., atol=1e-4)
        self.assertTrue(np.all(row['release'] >= -1e-8))
        self.assertTrue(np.all(row['release'] <= self.data.release_cap+1e-8))


if __name__ == '__main__':
    unittest.main(verbosity=2)
