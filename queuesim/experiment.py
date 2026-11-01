"""JSON-configured independent experiments and empirical waiting distributions."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np

from mm1.experiment import confidence_interval
from .distributions import Distribution
from .engine import Config, simulate, theory

METRICS = ('mean_number', 'mean_sojourn', 'mean_wait', 'mean_workload', 'utilization', 'little_residual')


def load_config(path=None):
    data = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    for name in ('arrival', 'service'):
        if name in data:
            data[name] = Distribution(**data[name])
    return Config(**data)


def run_experiment(config, replications=20, seed=20261102, allow_unstable=False):
    if type(replications) is not int or replications < 2:
        raise ValueError('At least two independent replications are required')
    if config.load >= 1 and not allow_unstable:
        raise ValueError('Load >= 1: use --allow-unstable for finite-horizon results only')
    sequences = np.random.SeedSequence(seed).spawn(replications)
    results = [simulate(config, sequence) for sequence in sequences]
    exact = theory(config)
    summary = []
    for metric in METRICS:
        values = [getattr(result, metric) for result in results]
        valid = sum(math.isfinite(v) for v in values)
        mean = low = high = se = None
        if valid == replications:
            mean, low, high, se = confidence_interval(values)
        target = exact.get(metric)
        summary.append(dict(metric=metric, estimate=mean, ci_low=low, ci_high=high,
                            standard_error=se, theory=target, valid_replications=valid,
                            total_replications=replications,
                            status='ok' if valid == replications else 'undefined: empty/censored/incomplete run'))
    return results, summary, sequences


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_results(output, config, results, summary, sequences, seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def finite(value):
        return value if math.isfinite(value) else None
    reps = []
    for index, (result, sequence) in enumerate(zip(results, sequences)):
        reps.append(dict(replication=index, root_seed=seed,
                         spawn_key='.'.join(map(str, sequence.spawn_key)),
                         **{key: finite(value) for key, value in result.metrics().items()},
                         arrivals=len(result.customers), censored_waits=result.censored_waits,
                         censored_sojourns=result.censored_sojourns,
                         observation_complete=result.observation_complete,
                         stop_reason=result.stop_reason, end_time=result.end_time))
    write_csv(output / 'replications.csv', list(reps[0]), reps)
    write_csv(output / 'summary.csv', list(summary[0]), summary)

    def customers():
        for index, result in enumerate(results):
            for c in result.customers:
                yield dict(replication=index, customer=c.identifier, arrival=c.arrival,
                           service=c.service, priority=c.priority, server=c.server,
                           start=c.start, departure=c.departure, wait=c.wait, sojourn=c.sojourn,
                           wait_censored=c.start is None, sojourn_censored=c.departure is None,
                           followup_until=result.end_time)
    fields = ('replication', 'customer', 'arrival', 'service', 'priority', 'server', 'start',
              'departure', 'wait', 'sojourn', 'wait_censored', 'sojourn_censored', 'followup_until')
    write_csv(output / 'customers.csv', fields, customers())
    # No completer-only CDF: all waiting outcomes and observation windows must be known.
    ecdf_available = all(r.observation_complete and r.censored_waits == 0 and r.customers
                         for r in results)
    waits = np.array([c.wait for r in results for c in r.customers]) if ecdf_available else np.array([])
    x, counts = np.unique(waits, return_counts=True)
    probabilities = np.cumsum(counts) / len(waits) if len(waits) else np.array([])
    write_csv(output / 'waiting_ecdf.csv', ('wait', 'cdf'),
              (dict(wait=float(v), cdf=float(p)) for v, p in zip(x, probabilities)))
    quantiles = [dict(probability=p, wait=float(np.quantile(waits, p, method='inverted_cdf')))
                 for p in (0.5, 0.9, 0.95, 0.99)] if len(waits) else []
    write_csv(output / 'waiting_quantiles.csv', ('probability', 'wait'), quantiles)
    metadata = dict(config=asdict(config), offered_load=config.load, seed=seed,
                    replications=len(results), mode='stable_load' if config.load < 1 else 'finite_horizon_only',
                    rng='PCG64; SeedSequence replication then arrival/service/priority children',
                    confidence='95% Student t on independent replication means; no dropping invalid runs',
                    waiting_ecdf_available=bool(ecdf_available),
                    waiting_ecdf_note='Pooled descriptive ECDF; correlated customers, no iid confidence band',
                    empty_or_censored_replications=[i for i, r in enumerate(results)
                                                   if not r.customers or r.censored_waits or not r.observation_complete],
                    python=platform.python_version(),
                    packages={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'matplotlib')},
                    source_sha256={str(p.relative_to(Path(__file__).parent.parent)):
                                   hashlib.sha256(p.read_bytes()).hexdigest()
                                   for pkg in ('queuesim', 'mm1')
                                   for p in (Path(__file__).parent.parent / pkg).glob('*.py')},
                    command=sys.argv)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    cache = output.resolve() / '.cache'
    for variable, directory in [('MPLCONFIGDIR', cache / 'matplotlib'), ('XDG_CACHE_HOME', cache)]:
        if variable not in os.environ:
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(directory)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if len(waits):
        # A left anchor at zero retains the waiting-time distribution's atom at zero.
        axes[0].step(np.r_[0, x], np.r_[0, probabilities], where='post')
        axes[1].hist(waits, bins=50, weights=np.ones(len(waits)) / len(waits))
        axes[1].set_ylabel('fraction per bin')
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'ECDF unavailable: empty, censored or incomplete run',
                    ha='center', va='center', transform=ax.transAxes, wrap=True)
    axes[0].set(ylabel='empirical P(wait <= x)', ylim=(0, 1.02))
    for ax in axes:
        ax.set_xlabel('waiting time')
        ax.grid(alpha=0.25)
    fig.suptitle(f'{config.discipline}, {config.servers} servers: pooled descriptive waiting distribution')
    fig.tight_layout()
    fig.savefig(output / 'waiting_distribution.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, row in zip(axes.flat, summary):
        ax.set_title(row['metric'])
        if row['estimate'] is not None:
            mean = row['estimate']
            ax.errorbar([0], [mean], yerr=[[mean - row['ci_low']], [row['ci_high'] - mean]], fmt='o', capsize=5)
        else:
            ax.text(0.5, 0.5, 'undefined', ha='center', transform=ax.transAxes)
        if row['theory'] is not None:
            ax.axhline(row['theory'], color='tab:red', linestyle='--')
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Independent replication means: 95% t CI; dashed = available FCFS theory')
    fig.tight_layout()
    fig.savefig(output / 'performance.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON Config; omitted: M/M/1 default')
    parser.add_argument('--replications', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--allow-unstable', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/general'))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        results, summary, sequences = run_experiment(config, args.replications, args.seed, args.allow_unstable)
        save_results(args.output, config, results, summary, sequences, args.seed)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(f'{config.arrival.name}/{config.service.name}/{config.servers}, '
          f'{config.discipline}, load={config.load:.6f}')
    if config.load >= 1:
        print('WARNING: finite-horizon only; no stationary interpretation')
    for row in summary:
        print(f"{row['metric']}: {row['estimate']}; 95% CI [{row['ci_low']}, {row['ci_high']}]; "
              f"theory={row['theory']}; {row['status']}")
    print(f'Censored waits: {sum(r.censored_waits for r in results)}; '
          f'censored sojourns: {sum(r.censored_sojourns for r in results)}')
    print(f'Outputs: {args.output}')
    return 0 if all(row['status'] == 'ok' for row in summary) else 1


if __name__ == '__main__':
    raise SystemExit(main())
