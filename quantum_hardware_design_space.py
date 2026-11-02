"""One-factor hardware sensitivity for a 3-pair distributed workload.

Uses the frozen adaptive-v3 controller and physics/state dynamics from
quantum_workload_equal_start.py; keep both files in the same directory.

Units: us for durations; ps for sensor error and wavepacket sigma;
Q in ps^2/us.  All settings are illustrative, not hardware predictions.

Example:
    python -u quantum_hardware_design_space.py --trials 100 --horizon 5000
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from quantum_workload_equal_start import Config, run_trial


def scenarios(base: Config, axis: str):
    # Explicit baselines let us compare one variable at a time.
    sweeps = {
        'sensor': ('sensor_std', [1.0, 2.0, 3.0, 4.0]),
        'measure': ('t_measure', [2.0, 5.0, 10.0]),
        'attempt': ('t_attempt', [5.0, 10.0, 20.0]),
        'drift': ('q', [0.01, 0.09, 0.20, 0.30, 0.40]),
    }
    groups = list(sweeps) if axis == 'all' else [axis]
    for group in groups:
        field, values = sweeps[group]
        for value in values:
            yield group, field, value, replace(base, **{field: value})


def describe(rows, cfg):
    times = np.array([r['completion_us'] for r in rows if r['finished']])
    n = len(rows)
    complete = len(times)
    success = sum(r['all_pairs_meet_fidelity'] for r in rows)
    if complete:
        # Quantiles conditional on completion; never infer deadline success from them.
        metrics = [float(np.mean(times))] + [float(np.percentile(times, x)) for x in (50, 95, 99)]
    else:
        metrics = [float('nan')] * 4
    deadline = min(1000., cfg.horizon)
    deadline_count = sum(bool(r['finished']) and r['completion_us'] <= deadline for r in rows)
    return {
        'completed': complete,
        'trials': n,
        'completion_pct': 100 * complete / n,
        'all_actual_good': success,
        'mean_us': metrics[0], 'p50_us': metrics[1],
        'p95_us': metrics[2], 'p99_us': metrics[3],
        'deadline_us': deadline, 'deadline_completion_pct': 100 * deadline_count / n,
        'mean_attempts': float(np.mean([r['attempts'] for r in rows])),
        'mean_measurements': float(np.mean([r['measurements'] for r in rows])),
        'mean_calibrations': float(np.mean([r['calibrations'] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trials', type=int, default=1000)
    ap.add_argument('--pairs', type=int, default=3)
    ap.add_argument('--axis', choices=['sensor', 'measure', 'attempt', 'drift', 'all'], default='all')
    ap.add_argument('--q', type=float, default=0.09, help='baseline Q in ps^2/us')
    ap.add_argument('--p-herald', type=float, default=0.25)
    ap.add_argument('--horizon', type=float, default=5000, help='us; incomplete trials censored')
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--output', type=Path, default=Path('hardware_design_summary.csv'))
    ap.add_argument('--trials-output', type=Path, default=None, help='optional raw-trial CSV')
    args = ap.parse_args()
    if args.trials < 1 or args.pairs < 1:
        ap.error('trials and pairs must be positive')
    baseline = Config(q=args.q, p_herald=args.p_herald, horizon=args.horizon)
    baseline.validate()
    configs = list(scenarios(baseline, args.axis))
    master = np.random.SeedSequence(args.seed)
    seeds = master.spawn(len(configs))
    summaries = []
    raw = []
    for (group, field, value, cfg), seed in zip(configs, seeds):
        cfg.validate()
        print(f'Running {group:7s} {field}={value:g} k={args.pairs} N={args.trials}', flush=True)
        trials = []
        for i, child in enumerate(seed.spawn(args.trials)):
            row = run_trial(cfg, args.pairs, 'adaptive', np.random.default_rng(child), 'equal')
            trials.append(row)
            if args.trials_output is not None:
                raw.append({'group': group, 'parameter': field, 'value': value,
                            'trial': i, **row})
        summary = {'group': group, 'parameter': field, 'value': value,
                   'q': cfg.q, 'sensor_std': cfg.sensor_std,
                   't_measure': cfg.t_measure, 't_attempt': cfg.t_attempt,
                   'pairs_required': args.pairs, **describe(trials, cfg)}
        summaries.append(summary)
        print(f'  completed={summary["completed"]}/{args.trials} '
              f'({summary["completion_pct"]:.1f}%) '
              f'by {summary["deadline_us"]:g}us={summary["deadline_completion_pct"]:.1f}% '
              f'mean={summary["mean_us"]:.1f}us '
              f'p95={summary["p95_us"]:.1f}us p99={summary["p99_us"]:.1f}us', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f'\nSummary saved to {args.output}', flush=True)
    if args.trials_output is not None:
        args.trials_output.parent.mkdir(parents=True, exist_ok=True)
        with args.trials_output.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(raw[0]))
            writer.writeheader()
            writer.writerows(raw)
        print(f'Raw trials saved to {args.trials_output}', flush=True)


if __name__ == '__main__':
    main()
