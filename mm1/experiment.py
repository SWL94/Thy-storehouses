"""Independent replications, Student t intervals, CSV and noninteractive figures."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.stats import t

from .simulation import Config, simulate, theory

METRICS = ('mean_wait', 'mean_sojourn', 'mean_number', 'utilization')


def confidence_interval(values):
    """95% two-sided t interval across independent replication estimates.

    This is an approximate interval for dependent within-run queue observations;
    no within-run customer independence is assumed. It does not remove bias.
    """
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError('At least two finite replication estimates are required; '
                         'increase duration if a replication has no observed arrivals')
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    half = float(t.ppf(0.975, len(x) - 1) * se)
    return mean, mean - half, mean + half, se


def run_experiment(config, replications=30, seed=20261102, *, allow_unstable=False):
    if replications < 2:
        raise ValueError('replications must be at least 2')
    if not config.stable and not allow_unstable:
        raise ValueError('Unstable/critical queue: use --allow-unstable for finite-horizon estimates')
    seeds = np.random.SeedSequence(seed).spawn(replications)
    results = [simulate(config, s) for s in seeds]
    exact = theory(config) if config.stable else {}
    summary = []
    for metric in (*METRICS, 'little_residual'):
        mean, low, high, se = confidence_interval([getattr(r, metric) for r in results])
        target = exact.get(metric)
        summary.append(dict(metric=metric, estimate=mean, ci_low=low, ci_high=high,
                            standard_error=se, theory=target,
                            relative_error=(mean / target - 1) if target else None,
                            theory_in_ci=(low <= target <= high) if target is not None else None))
    return results, summary, seeds


def save_results(output, config, results, summary, seeds, seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (result, sequence) in enumerate(zip(results, seeds)):
        row = {k: v for k, v in asdict(result).items() if k != 'trace'}
        row.update(replication=index, root_seed=seed,
                   spawn_key='.'.join(map(str, sequence.spawn_key)))
        rows.append(row)
    for name, data in [('replications.csv', rows), ('summary.csv', summary)]:
        with (output / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    metadata = dict(config=asdict(config), root_seed=seed, replications=len(results),
                    mode='stationary_comparison' if config.stable else 'finite_horizon_only',
                    rng='NumPy default_rng (PCG64), SeedSequence replication and arrival/service children',
                    confidence='95% Student t across independent replication means',
                    python=platform.python_version(),
                    packages={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'matplotlib')},
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob('*.py')},
                    command=sys.argv)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    # Cloud home directories can be read-only. Respect user-supplied cache paths.
    cache = output.resolve() / '.cache'
    for variable, directory in [('MPLCONFIGDIR', cache / 'matplotlib'),
                                ('XDG_CACHE_HOME', cache)]:
        if variable not in os.environ:
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(directory)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, row in zip(axes.flat, summary[:4]):
        mean = row['estimate']
        ax.errorbar([0], [mean], yerr=[[mean - row['ci_low']], [row['ci_high'] - mean]],
                    fmt='o', capsize=6, label='simulation: 95% t CI')
        if row['theory'] is not None:
            ax.axhline(row['theory'], color='tab:red', linestyle='--', label='stationary theory')
        ax.set_title(row['metric'])
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'M/M/1: lambda={config.arrival_rate}, mu={config.service_rate}; '
                 + ('stationary comparison' if config.stable else 'FINITE HORIZON (no stationary theory)'))
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.flat, METRICS):
        values = np.array([getattr(r, metric) for r in results])
        ax.plot(np.arange(1, len(values) + 1), np.cumsum(values) / np.arange(1, len(values) + 1))
        target = next(row['theory'] for row in summary if row['metric'] == metric)
        if target is not None:
            ax.axhline(target, color='tab:red', linestyle='--')
        ax.set(title=metric, xlabel='independent replications accumulated')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / 'replication_convergence.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrival-rate', type=float, default=0.8)
    parser.add_argument('--service-rate', type=float, default=1.0)
    parser.add_argument('--warmup', type=float, default=2000)
    parser.add_argument('--duration', type=float, default=20000)
    parser.add_argument('--replications', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--allow-unstable', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/baseline'))
    args = parser.parse_args(argv)
    try:
        config = Config(args.arrival_rate, args.service_rate, args.warmup, args.duration)
        results, summary, seeds = run_experiment(config, args.replications, args.seed,
                                                allow_unstable=args.allow_unstable)
        save_results(args.output, config, results, summary, seeds, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if not config.stable:
        print('WARNING: finite-horizon estimates only; warmup does not create stationarity.')
    for row in summary:
        print(f"{row['metric']:16} {row['estimate']:.6f} "
              f"95% CI [{row['ci_low']:.6f}, {row['ci_high']:.6f}] theory={row['theory']}")
    print(f'Outputs: {args.output}')


if __name__ == '__main__':
    main()
