import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from queuesim import Config, Distribution, simulate, theory
from queuesim.experiment import run_experiment


def draws(values):
    return iter(values).__next__


def fixed(**kwargs):
    return Config(warmup=0, duration=8, followup=20, **kwargs)


class GeneralEngineTests(unittest.TestCase):
    def test_single_server_workload_integral(self):
        r = simulate(fixed(), arrival_draw=draws([1, 1, math.inf]),
                     service_draw=draws([3, 2]), priority_draw=lambda: 0)
        self.assertEqual(r.mean_wait, 1)
        self.assertEqual(r.mean_sojourn, 3.5)
        self.assertEqual(r.mean_number, 7 / 8)
        self.assertEqual(r.mean_workload, 10.5 / 8)
        self.assertEqual(r.utilization, 5 / 8)
        self.assertEqual(r.little_residual, 0)

    def test_multiple_servers_and_warmup(self):
        for warmup, duration, number, work, util in [(0, 7, 10/7, 15/7, 4/7),
                                                    (2, 3, 8/3, 11/3, 1)]:
            with self.subTest(warmup=warmup):
                r = simulate(Config(servers=2, warmup=warmup, duration=duration),
                             record_trace=True, arrival_draw=draws([1, 1, 1, math.inf]),
                             service_draw=draws([4, 3, 1]), priority_draw=lambda: 0)
                self.assertAlmostEqual(r.mean_number, number)
                self.assertAlmostEqual(r.mean_workload, work)
                self.assertAlmostEqual(r.utilization, util)
                self.assertEqual([(t, cid) for t, kind, cid, _ in r.trace if kind == 'service_start'],
                                 [(1, 0), (2, 1), (5, 2)])
                self.assertEqual(len({c.server for c in r.customers}), 2)

    def test_fcfs_lcfs_priority_sjf_selection(self):
        expected = {'FCFS': [0, 1, 2, 3], 'LCFS': [0, 3, 2, 1],
                    'PR': [0, 2, 3, 1], 'SJF': [0, 3, 2, 1]}
        for discipline, order in expected.items():
            with self.subTest(discipline=discipline):
                r = simulate(fixed(discipline=discipline), record_trace=True,
                             arrival_draw=draws([1, 1, 1, 1, math.inf]),
                             service_draw=draws([4, 3, 2, 1]),
                             priority_draw=draws([2, 2, 0, 1]))
                self.assertEqual([cid for _, kind, cid, _ in r.trace if kind == 'service_start'], order)
                self.assertEqual(r.customers[0].departure, 5)  # no preemption

    def test_priority_ties_are_fcfs(self):
        r = simulate(fixed(discipline='PR'), record_trace=True,
                     arrival_draw=draws([1, 1, 1, math.inf]),
                     service_draw=draws([4, 1, 1]), priority_draw=lambda: 0)
        self.assertEqual([cid for _, kind, cid, _ in r.trace if kind == 'service_start'], [0, 1, 2])

    def test_post_window_arrivals_delay_lcfs_cohort(self):
        r = simulate(Config(discipline='LCFS', warmup=0, duration=3.5, followup=20),
                     arrival_draw=draws([1, 1, 1, 1, math.inf]),
                     service_draw=draws([4, 1, 1, 1]), priority_draw=lambda: 0)
        self.assertEqual(len(r.customers), 3)
        self.assertEqual([c.start for c in r.customers], [1, 7, 6])
        self.assertEqual(r.mean_wait, 8 / 3)
        self.assertEqual(r.end_time, 8)
        self.assertEqual(r.censored_sojourns, 0)

    def test_censoring_does_not_drop_incomplete_customers(self):
        r = simulate(Config(warmup=0, duration=3, followup=0),
                     arrival_draw=draws([1, 1, math.inf]),
                     service_draw=draws([10, 1]), priority_draw=lambda: 0)
        self.assertEqual(r.censored_waits, 1)
        self.assertEqual(r.censored_sojourns, 2)
        self.assertTrue(math.isnan(r.mean_wait))
        self.assertTrue(math.isnan(r.mean_sojourn))
        self.assertEqual(r.end_time, 3)
        self.assertTrue(r.observation_complete)
        self.assertEqual(r.stop_reason, 'followup_limit')
        self.assertAlmostEqual(r.mean_number, 1)
        self.assertAlmostEqual(r.mean_workload, 19 / 3)

    def test_empty_boundary_and_event_guard(self):
        empty = simulate(fixed(), arrival_draw=lambda: math.inf)
        self.assertEqual(empty.mean_number, 0)
        self.assertEqual(empty.mean_workload, 0)
        self.assertTrue(math.isnan(empty.mean_wait))
        boundary = simulate(Config(warmup=0, duration=2), arrival_draw=lambda: 1,
                            service_draw=lambda: 0.5, priority_draw=lambda: 0)
        self.assertEqual(len(boundary.customers), 1)  # excludes t=2 arrival
        limited = simulate(fixed(max_events=1), arrival_draw=lambda: 1,
                           service_draw=lambda: 2, priority_draw=lambda: 0)
        self.assertFalse(limited.observation_complete)
        self.assertTrue(math.isnan(limited.mean_number))
        self.assertTrue(math.isnan(limited.mean_wait))
        self.assertEqual(limited.stop_reason, 'max_events')

    def test_simultaneous_completion_precedes_arrival(self):
        r = simulate(Config(warmup=0, duration=3), record_trace=True,
                     arrival_draw=draws([1, 1, math.inf]),
                     service_draw=lambda: 1, priority_draw=lambda: 0)
        self.assertEqual([kind for t, kind, _, _ in r.trace if t == 2],
                         ['departure', 'arrival', 'service_start'])

    def test_reproducible_streams(self):
        a = simulate(fixed(), 42, record_trace=True)
        b = simulate(fixed(), 42, record_trace=True)
        self.assertEqual(a, b)
        self.assertNotEqual(a.trace, simulate(fixed(), 43, record_trace=True).trace)

    def test_validation_and_unstable(self):
        for kwargs in (dict(servers=0), dict(servers=1.5), dict(discipline='bad'),
                       dict(followup=-1), dict(duration=0), dict(max_events=0),
                       dict(priority_probabilities=(0.2, 0.3))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)
        for value in (0, -1, math.nan):
            with self.subTest(draw=value), self.assertRaises(ValueError):
                simulate(fixed(), arrival_draw=lambda: value)
        with self.assertRaises(ValueError):
            simulate(fixed(), arrival_draw=lambda: 1, service_draw=lambda: math.inf)
        overloaded = Config(arrival=Distribution('exponential', {'mean': 0.5}),
                            warmup=0, duration=10)
        self.assertEqual(theory(overloaded), {})
        with self.assertRaises(ValueError):
            run_experiment(overloaded, 2)
        results, _, _ = run_experiment(overloaded, 2, allow_unstable=True)
        self.assertTrue(all(r.observation_complete for r in results))


class DistributionAndTheoryTests(unittest.TestCase):
    def test_distribution_moments_and_samples(self):
        specs = [('exponential', {'mean': 2}, 2, 8),
                 ('gamma', {'shape': 2, 'scale': 3}, 6, 54),
                 ('erlang', {'shape': 3, 'scale': 2}, 6, 48),
                 ('weibull', {'shape': 1, 'scale': 2}, 2, 8),
                 ('lognormal', {'meanlog': 0, 'sigma': 1}, math.exp(.5), math.exp(2)),
                 ('uniform', {'low': 1, 'high': 3}, 2, 13/3),
                 ('deterministic', {'value': 2}, 2, 4)]
        for name, parameters, mean, second in specs:
            with self.subTest(name=name):
                d = Distribution(name, parameters)
                self.assertAlmostEqual(d.mean, mean)
                self.assertAlmostEqual(d.second_moment, second)
                rng = np.random.default_rng(987)
                x = np.array([d.sample(rng) for _ in range(30000)])
                self.assertTrue(np.all(x > 0))
                self.assertLess(abs(x.mean() - mean), 6 * x.std() / np.sqrt(len(x)) + .001)
        for name, p in [('unknown', {}), ('gamma', {'shape': 0, 'scale': 1}),
                        ('erlang', {'shape': 1.5, 'scale': 1}),
                        ('uniform', {'low': 2, 'high': 1}),
                        ('lognormal', {'meanlog': 0, 'sigma': -1}),
                        ('exponential', {'mean': math.inf})]:
            with self.subTest(name=name, p=p), self.assertRaises(ValueError):
                Distribution(name, p)

    def test_exact_special_cases(self):
        mm1 = theory(Config())
        self.assertAlmostEqual(mm1['mean_wait'], 4)
        self.assertAlmostEqual(mm1['mean_workload'], 4)
        md1 = theory(Config(service=Distribution('deterministic', {'value': 1})))
        self.assertAlmostEqual(md1['mean_wait'], 2)
        self.assertAlmostEqual(md1['mean_workload'], 2)
        mm2 = theory(Config(arrival=Distribution('exponential', {'mean': 0.625}), servers=2))
        self.assertAlmostEqual(mm2['mean_wait'], 16 / 9)
        self.assertAlmostEqual(mm2['mean_number'], 40 / 9)
        self.assertEqual(theory(Config(discipline='PR')), {})
        self.assertEqual(theory(Config(servers=2, service=Distribution('gamma', {'shape': 2, 'scale': 1}))), {})

    def test_long_run_mgc_and_multiserver(self):
        for config in (Config(service=Distribution('deterministic', {'value': 1}),
                              warmup=1000, duration=10000),
                       Config(arrival=Distribution('exponential', {'mean': .625}), servers=2,
                              warmup=1000, duration=10000)):
            _, summary, _ = run_experiment(config, 20, 12345)
            for row in summary[:-1]:
                with self.subTest(servers=config.servers, metric=row['metric']):
                    self.assertEqual(row['status'], 'ok')
                    self.assertLess(abs(row['estimate'] - row['theory']),
                                    5 * row['standard_error'] + .02 * row['theory'])
            little = summary[-1]
            self.assertLess(abs(little['estimate']), 5 * little['standard_error'] + .01)


class GeneralCLITests(unittest.TestCase):
    def test_outputs_ecdf_quantiles_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tmp / 'config.json'
            config.write_text(json.dumps(dict(arrival=dict(name='gamma', parameters=dict(shape=2, scale=.5)),
                                             servers=2, warmup=20, duration=400, followup=100)))
            for index in ('a', 'b'):
                proc = subprocess.run([sys.executable, '-m', 'queuesim', '--config', str(config),
                                       '--replications', '3', '--seed', '42', '--output', str(tmp / index)],
                                      capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
                self.assertEqual((tmp / index / 'waiting_distribution.png').read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
                self.assertTrue(json.loads((tmp / index / 'metadata.json').read_text())['waiting_ecdf_available'])
            for filename in ('replications.csv', 'customers.csv', 'summary.csv', 'waiting_ecdf.csv', 'waiting_quantiles.csv'):
                self.assertEqual((tmp / 'a' / filename).read_bytes(), (tmp / 'b' / filename).read_bytes())
            with (tmp / 'a' / 'waiting_ecdf.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(float(rows[-1]['cdf']), 1)
            self.assertEqual(float(rows[0]['wait']), 0)
            with (tmp / 'a' / 'customers.csv').open() as f:
                waits = np.array([float(row['wait']) for row in csv.DictReader(f)])
            self.assertAlmostEqual(float(rows[0]['cdf']), np.mean(waits == 0))
            with (tmp / 'a' / 'waiting_quantiles.csv').open() as f:
                for row in csv.DictReader(f):
                    self.assertEqual(float(row['wait']), np.quantile(waits, float(row['probability']), method='inverted_cdf'))

    def test_censored_cli_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tmp / 'censored.json'
            config.write_text(json.dumps(dict(arrival=dict(name='deterministic', parameters=dict(value=1)),
                                             service=dict(name='deterministic', parameters=dict(value=10)),
                                             warmup=0, duration=3, followup=0)))
            proc = subprocess.run([sys.executable, '-m', 'queuesim', '--config', str(config),
                                   '--replications', '2', '--allow-unstable', '--output', str(tmp / 'out')],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertIn('undefined', proc.stdout)
            meta = json.loads((tmp / 'out' / 'metadata.json').read_text())
            self.assertFalse(meta['waiting_ecdf_available'])
            with (tmp / 'out' / 'waiting_ecdf.csv').open() as f:
                self.assertEqual(list(csv.DictReader(f)), [])
