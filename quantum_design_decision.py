"""Nominal-rate vs uncertainty-aware photonic interconnect design selection.

Uses the frozen adaptive-v3 link controller from quantum_workload_equal_start.py.
Copy both scripts into the same directory. Requires numpy.

Designs are HYPOTHETICAL hardware tradeoffs, not measured components:
  fast:     shorter attempts, less photon detection success
  balanced: longer attempts, greater photon detection success
The 'rate_only' benchmark is an intentionally simplified architecture abstraction,
NOT a claimed state-of-the-art synchronization-aware baseline.

Each candidate is evaluated on a separate training and held-out evaluation sample.
The uncertainty-aware selector maximizes the training deadline-completion
probability; ties are broken by smaller capped completion time.

No true fidelity information is used to select successful heralds, and no
uncompleted run is dropped when scoring a design.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from quantum_workload_equal_start import Config, run_trial


@dataclass(frozen=True)
class Hardware:
    name: str
    attempt_us: float
    herald_probability: float

    @property
    def nominal_pairs_per_second(self):
        return 1_000_000 * self.herald_probability / self.attempt_us


CANDIDATES = (
    Hardware('fast_low_collection', 5.0, 0.10),
    Hardware('balanced_collection', 10.0, 0.25),
)


def evaluate(hw, base, n, k, deadline, seeds):
    cfg = replace(base, t_attempt=hw.attempt_us, p_herald=hw.herald_probability)
    cfg.validate()
    values = []
    good = 0
    deadline_hits = 0
    completed = 0
    total_measure = total_cal = total_attempts = 0
    raw = []
    for i, ss in enumerate(seeds):
        result = run_trial(cfg, k, 'adaptive', np.random.default_rng(ss), 'equal')
        done = bool(result['finished'])
        lat = float(result['completion_us']) if done else float('nan')
        deadline_ok = done and lat <= deadline
        good += result['all_pairs_meet_fidelity']
        completed += int(done)
        deadline_hits += int(deadline_ok)
        values.append(lat)
        total_measure += result['measurements']
        total_cal += result['calibrations']
        total_attempts += result['attempts']
        raw.append({'candidate': hw.name, 'trial': i, 'finished': int(done),
                    'deadline_hit': int(deadline_ok), 'completion_us': lat,
                    'all_actual_good': result['all_pairs_meet_fidelity'],
                    'attempts': result['attempts'],
                    'measurements': result['measurements'],
                    'calibrations': result['calibrations']})
    arr = np.array(values, dtype=float)
    observed = arr[np.isfinite(arr)]
    capped = np.where(np.isfinite(arr), arr, base.horizon)
    summary = {'candidate': hw.name, 'n': n, 'q': base.q,
        'attempt_us': hw.attempt_us, 'p_herald': hw.herald_probability,
        'nominal_pairs_per_s': hw.nominal_pairs_per_second,
        'completed': completed, 'deadline_hits': deadline_hits,
        'deadline_pct': 100*deadline_hits/n,
        'all_good_pct': 100*good/n,
        'mean_completed_us': float(observed.mean()) if len(observed) else float('nan'),
        'p95_completed_us': float(np.percentile(observed,95)) if len(observed) else float('nan'),
        'p99_completed_us': float(np.percentile(observed,99)) if len(observed) else float('nan'),
        'mean_capped_us': float(capped.mean()),
        'mean_attempts': total_attempts/n,
        'mean_measurements': total_measure/n,
        'mean_calibrations': total_cal/n}
    return summary, raw


def choose_uncertainty_aware(training):
    return max(training, key=lambda r: (r['deadline_pct'], -r['mean_capped_us']))['candidate']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=int, default=300, help='trials per design for selecting hardware')
    ap.add_argument('--eval', type=int, default=1000, help='held-out trials per design')
    ap.add_argument('--q', type=float, nargs='+', default=[0.09,0.30,0.40])
    ap.add_argument('--pairs', type=int, default=3)
    ap.add_argument('--deadline', type=float, default=1000.0)
    ap.add_argument('--horizon', type=float, default=5000.0)
    ap.add_argument('--seed', type=int, default=20261103)
    ap.add_argument('--output', type=Path, default=Path('design_decision_summary.csv'))
    ap.add_argument('--raw-output', type=Path, default=None)
    args = ap.parse_args()
    if args.train < 1 or args.eval < 1 or args.pairs < 1 or args.deadline <= 0 or args.horizon < args.deadline:
        ap.error('Use positive counts and 0 < deadline <= horizon')
    nominal_pick = max(CANDIDATES, key=lambda h: h.nominal_pairs_per_second).name
    print('HYPOTHETICAL designs; nominal rate ignores maintenance and timing feasibility.', flush=True)
    print('Nominal-rate selector chooses:',nominal_pick, flush=True)
    rows, raw_rows = [], []
    root = np.random.SeedSequence(args.seed)
    q_seeds = root.spawn(len(args.q))
    for q, q_seed in zip(args.q, q_seeds):
        base = Config(q=q, horizon=args.horizon)
        base.validate()
        # Deliberately separate training and evaluation streams. Per-design
        # independent streams prevent accidental RNG consumption coupling.
        streams = q_seed.spawn(2*len(CANDIDATES))
        print(f'\nQ={q:g}, workload={args.pairs} pairs, deadline={args.deadline:g} us', flush=True)
        train_rows, eval_rows = [], []
        for j, hw in enumerate(CANDIDATES):
            print(f'  Training {hw.name} ({args.train} trials)', flush=True)
            train_summary, _ = evaluate(hw, base, args.train, args.pairs, args.deadline,
                                        streams[j].spawn(args.train))
            train_rows.append(train_summary)
        aware_pick = choose_uncertainty_aware(train_rows)
        print('  Uncertainty-aware selector chooses:',aware_pick, flush=True)
        for j, hw in enumerate(CANDIDATES):
            print(f'  Evaluating {hw.name} ({args.eval} HELD-OUT trials)', flush=True)
            summary, raw = evaluate(hw, base, args.eval, args.pairs, args.deadline,
                                    streams[len(CANDIDATES)+j].spawn(args.eval))
            summary['selected_by_rate_only'] = int(hw.name == nominal_pick)
            summary['selected_by_uncertainty_aware'] = int(hw.name == aware_pick)
            summary['training_deadline_pct'] = train_rows[j]['deadline_pct']
            eval_rows.append(summary)
            for r in raw:
                r.update({'q': q, 'attempt_us':hw.attempt_us,'p_herald':hw.herald_probability})
            if args.raw_output is not None:
                raw_rows.extend(raw)
            print(f'    deadline={summary["deadline_pct"]:.1f}% '
                  f'completed={summary["completed"]}/{args.eval} '
                  f'mean(completed)={summary["mean_completed_us"]:.1f} us '
                  f'p95(completed)={summary["p95_completed_us"]:.1f} us '
                  f'measures={summary["mean_measurements"]:.1f} '
                  f'calibrations={summary["mean_calibrations"]:.1f}', flush=True)
        rows.extend(eval_rows)
        by_name = {r['candidate']:r for r in eval_rows}
        d = by_name[aware_pick]['deadline_pct'] - by_name[nominal_pick]['deadline_pct']
        print(f'  HELD-OUT selection difference: {d:+.1f} percentage points '
              f'(aware minus rate-only)', flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('\nSummary saved to', args.output,flush=True)
    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True,exist_ok=True)
        with args.raw_output.open('w', newline='', encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(raw_rows[0]));writer.writeheader();writer.writerows(raw_rows)
        print('Raw trials saved to', args.raw_output,flush=True)

if __name__ == '__main__':
    main()
