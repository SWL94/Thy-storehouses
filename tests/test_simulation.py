import math
import unittest

import numpy as np

from mm1 import Config, simulate, theory
from mm1.experiment import confidence_interval, run_experiment


def draws(values):
    return iter(values).__next__


class DeterministicTests(unittest.TestCase):
    def test_empty_system(self):
        result = simulate(Config(warmup=0, duration=5), record_trace=True,
                          arrival_draw=lambda: math.inf)
        self.assertEqual(result.arrivals, 0)
        self.assertTrue(math.isnan(result.mean_wait))
        self.assertEqual(result.mean_number, 0)
        self.assertEqual(result.utilization, 0)
        self.assertEqual(result.trace, [])

    def test_fcfs_events_and_exact_areas(self):
        # arrivals 1,2,3; services [3,2,1]; starts 1,4,6; exits 4,6,7.
        result = simulate(Config(warmup=0, duration=8), record_trace=True,
                          arrival_draw=draws([1, 1, 1, math.inf]),
                          service_draw=draws([3, 2, 1]))
        self.assertEqual([(e.customer, e.time) for e in result.trace if e.kind == 'arrival'],
                         [(0, 1), (1, 2), (2, 3)])
        self.assertEqual([(e.customer, e.time) for e in result.trace if e.kind == 'service_start'],
                         [(0, 1), (1, 4), (2, 6)])
        self.assertEqual([(e.customer, e.time) for e in result.trace if e.kind == 'departure'],
                         [(0, 4), (1, 6), (2, 7)])
        self.assertAlmostEqual(result.mean_wait, 5 / 3)
        self.assertAlmostEqual(result.mean_sojourn, 11 / 3)
        self.assertAlmostEqual(result.mean_number, 11 / 8)
        self.assertAlmostEqual(result.utilization, 6 / 8)
        self.assertAlmostEqual(result.little_residual, 0)

    def test_warmup_clipping_and_cohort_drain(self):
        # Customer 0 arrives before warmup but contributes to time integrals.
        # Customer 1 arrives at warmup and exits after the observation ends.
        result = simulate(Config(warmup=2, duration=2), record_trace=True,
                          arrival_draw=draws([1, 1, math.inf]),
                          service_draw=draws([3, 2]))
        self.assertEqual(result.arrivals, 1)
        self.assertEqual(result.mean_wait, 2)
        self.assertEqual(result.mean_sojourn, 4)
        self.assertEqual(result.mean_number, 2)
        self.assertEqual(result.utilization, 1)
        self.assertEqual(result.end_time, 6)
        self.assertAlmostEqual(result.little_residual, 0)

    def test_observation_end_is_excluded(self):
        r = simulate(Config(warmup=0, duration=2), arrival_draw=draws([1, 1]),
                     service_draw=lambda: 3)
        self.assertEqual(r.arrivals, 1)
        self.assertEqual(r.mean_sojourn, 3)
        self.assertEqual(r.mean_number, 0.5)
        self.assertEqual(r.utilization, 0.5)
        self.assertEqual(r.end_time, 4)
        self.assertNotEqual(r.little_residual, 0)  # finite-window boundary effect

    def test_simultaneous_departure_before_arrival(self):
        r = simulate(Config(warmup=0, duration=3), record_trace=True,
                     arrival_draw=draws([1, 1, math.inf]), service_draw=lambda: 1)
        self.assertEqual([(e.kind, e.customer) for e in r.trace if e.time == 2],
                         [('departure', 0), ('arrival', 1), ('service_start', 1)])
        self.assertEqual(r.mean_wait, 0)

    def test_seed_reproducibility_and_independence(self):
        config = Config(warmup=10, duration=200)
        a = simulate(config, 123, record_trace=True)
        self.assertEqual(a, simulate(config, 123, record_trace=True))
        self.assertNotEqual(a.trace, simulate(config, 124, record_trace=True).trace)
        children = np.random.SeedSequence(123).spawn(2)
        self.assertNotEqual(simulate(config, children[0]).metrics(),
                            simulate(config, children[1]).metrics())

    def test_invalid_config_and_draws(self):
        for kwargs in [dict(arrival_rate=0), dict(service_rate=-1), dict(warmup=-1),
                       dict(duration=0), dict(duration=math.inf), dict(warmup=math.nan)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)
        with self.assertRaises(ValueError):
            simulate(Config(), arrival_draw=lambda: 0)
        with self.assertRaises(ValueError):
            simulate(Config(), arrival_draw=lambda: 1, service_draw=lambda: math.inf)

    def test_unstable_handling(self):
        for lam in (1.0, 1.2):
            c = Config(lam, 1, 0, 100)
            with self.assertRaises(ValueError):
                theory(c)
            with self.assertRaises(ValueError):
                run_experiment(c, 2)
            results, rows, _ = run_experiment(c, 3, allow_unstable=True)
            self.assertTrue(all(row['theory'] is None for row in rows))
            self.assertTrue(all(0 <= r.utilization <= 1 for r in results))

    def test_student_t_interval(self):
        mean, low, high, se = confidence_interval([1, 2, 3, 4, 5])
        self.assertEqual(mean, 3)
        self.assertAlmostEqual(se, math.sqrt(0.5))
        self.assertAlmostEqual(high - mean, 2.776445105 * se, places=7)
        self.assertAlmostEqual(mean - low, high - mean)
        for x in ([1], [1, math.nan]):
            with self.assertRaises(ValueError):
                confidence_interval(x)


class StatisticalTests(unittest.TestCase):
    def test_long_run_theory_and_little_law(self):
        # A 95% CI need not cover theory in every valid experiment. Use a
        # conservative 5-SE + 1%-scale tolerance to diagnose material errors.
        c = Config(0.8, 1, 2000, 30000)
        results, summary, _ = run_experiment(c, 32, 314159)
        for row in summary[:4]:
            with self.subTest(metric=row['metric']):
                self.assertLess(abs(row['estimate'] - row['theory']),
                                5 * row['standard_error'] + 0.01 * row['theory'])
        residuals = np.array([r.little_residual for r in results])
        _, _, _, se = confidence_interval(residuals)
        self.assertLess(abs(residuals.mean()), 5 * se + 0.01)
        self.assertLess(np.mean(np.abs(residuals)), 0.02)
        # Each matched-cohort L=lambda_hat*W agrees with integrated L up to
        # finite-window customer contributions; it is not an exact identity.

    def test_longer_runs_reduce_sampling_variation(self):
        exact = theory(Config())['mean_sojourn']
        short, _, _ = run_experiment(Config(warmup=2000, duration=300), 40, 271828)
        long, _, _ = run_experiment(Config(warmup=2000, duration=15000), 40, 271828)
        short_rmse = np.sqrt(np.mean([(r.mean_sojourn - exact)**2 for r in short]))
        long_rmse = np.sqrt(np.mean([(r.mean_sojourn - exact)**2 for r in long]))
        self.assertLess(long_rmse, short_rmse * 0.7)


if __name__ == '__main__':
    unittest.main()
