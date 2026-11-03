"""Gaussian versus one-sided-exponential photon envelope sensitivity.

Uses the SAME drift, sensing, calibration, adaptive-v3 admission,
and herald-driven completion logic as quantum_workload_equal_start.py.
Only the optical fidelity law and its equivalent timing acceptance threshold change.

Units: timing offsets ps; duration us; Q ps^2/us.
The exponential model assumes a normalized causal amplitude f(t) proportional to
exp(-t/(2*tau)) 1_{t>=0}, intensity standard deviation tau.
Therefore squared photon overlap is exp(-abs(delta)/tau).
The Gaussian model uses intensity standard deviation sigma.

These are idealized, unfiltered, polarization-resolved heralds. No source
chirp, detector window, non-timing infidelity, or finite-efficiency calibration.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import quantum_workload_equal_start as sim


def timing_limit(shape: str, width_ps: float, fmin: float) -> float:
    if fmin <= 0.5:
        return math.inf
    c = -math.log(2 * fmin - 1)
    return (2 * width_ps * math.sqrt(c)) if shape == 'gaussian' else (width_ps * c)


def fidelity(shape: str, delta: float, width_ps: float) -> float:
    exponent = -(delta**2 / (4 * width_ps**2)) if shape == 'gaussian' else -abs(delta) / width_ps
    return 0.5 * (1.0 + math.exp(exponent))


def run_setting(shape: str, width_ps: float, q: float, trials: int, pairs: int,
                p_herald: float, horizon: float, seed: int) -> list[dict]:
    cfg = sim.Config(q=q, sigma=width_ps, p_herald=p_herald, horizon=horizon)
    cfg.validate()
    threshold = timing_limit(shape, width_ps, cfg.min_fidelity)
    # Important: Even with zero belief variance and perfectly centered mean,
    # random drift during a full attempt produces irreducible variance Q*t_a.
    def reliable(st: sim.State, local_cfg: sim.Config) -> float:
        var = st.v + local_cfg.q * local_cfg.t_attempt
        if var <= 0:
            return float(abs(st.m) <= threshold)
        sd = math.sqrt(var)
        return max(0.0, min(1.0,
            sim.normal_cdf((threshold - st.m) / sd) -
            sim.normal_cdf((-threshold - st.m) / sd)))
    def optical_fidelity(delta: float, local_cfg: sim.Config) -> float:
        return fidelity(shape, delta, width_ps)
    # Monkeypatch scoped to a purely sequential driver; all other simulator
    # operations remain original. Restore in finally even after exceptions.
    orig_reliability = sim.reliability_after_attempt
    orig_fidelity = sim.fidelity
    sim.reliability_after_attempt = reliable
    sim.fidelity = optical_fidelity
    try:
        seq = np.random.SeedSequence(seed).spawn(trials)
        rows = []
        # For impossible settings, prove the max admissibility probability is
        # below the threshold. Avoid burning thousands of futile maintenance steps.
        max_p = 1.0 if cfg.q == 0 else (
            2 * sim.normal_cdf(threshold / math.sqrt(cfg.q * cfg.t_attempt)) - 1)
        impossible = max_p + 1e-12 < cfg.min_reliability
        for i, s in enumerate(seq):
            if impossible:
                result = dict(finished=0, completion_us=math.nan,
                              observed_us=0.0, attempts=0, measurements=0,
                              calibrations=0, all_pairs_meet_fidelity=0,
                              infeasible_gate=1)
            else:
                result = sim.run_trial(cfg, pairs, 'adaptive', np.random.default_rng(s), 'equal')
                result['infeasible_gate'] = 0
            result.update(shape=shape, width_ps=width_ps, q=q, trial=i,
                          delta_threshold_ps=threshold,
                          max_possible_admission_p=max_p)
            rows.append(result)
    finally:
        sim.reliability_after_attempt = orig_reliability
        sim.fidelity = orig_fidelity
    return rows


def summarize(rows: list[dict], deadline: float) -> dict:
    durations = np.array([r['completion_us'] for r in rows if r['finished']], float)
    n = len(rows)
    hits = sum(r['finished'] and r['completion_us'] <= deadline for r in rows)
    return dict(shape=rows[0]['shape'], width_ps=rows[0]['width_ps'], q=rows[0]['q'],
                n=n, finished=int(len(durations)), deadline_hits=hits,
                deadline_pct=100.0 * hits / n,
                mean_us=float(np.mean(durations)) if len(durations) else math.nan,
                p95_us=float(np.percentile(durations, 95)) if len(durations) else math.nan,
                p99_us=float(np.percentile(durations, 99)) if len(durations) else math.nan,
                attempts=float(np.mean([r['attempts'] for r in rows])),
                measurements=float(np.mean([r['measurements'] for r in rows])),
                calibrations=float(np.mean([r['calibrations'] for r in rows])),
                max_possible_admission_p=rows[0]['max_possible_admission_p'],
                delta_threshold_ps=rows[0]['delta_threshold_ps'],
                infeasible_gate=rows[0]['infeasible_gate'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trials', type=int, default=1000)
    ap.add_argument('--q', nargs='+', type=float, default=[0.001, 0.01, 0.03, 0.09])
    ap.add_argument('--widths', nargs='+', type=float, default=[10.0])
    ap.add_argument('--pairs', type=int, default=3)
    ap.add_argument('--p-herald', type=float, default=0.25)
    ap.add_argument('--deadline', type=float, default=1000.0)
    ap.add_argument('--horizon', type=float, default=5000.0)
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--output', type=Path, default=Path('wavepacket_robustness.csv'))
    args = ap.parse_args()
    if args.trials < 1 or args.pairs < 1 or min(args.widths) <= 0 or min(args.q) < 0:
        ap.error('Need positive trials, pairs, widths, and nonnegative Q')
    if args.deadline > args.horizon:
        ap.error('--deadline must not exceed --horizon')
    summaries = []
    for width in args.widths:
        for q in args.q:
            for shape in ('gaussian', 'exponential'):
                print(f'Running {shape:11s} width={width:g}ps Q={q:g} N={args.trials}', flush=True)
                # Same seed used for each setting; random-number consumption
                # can differ, so these are not guaranteed coupled trials.
                rows = run_setting(shape, width, q, args.trials, args.pairs,
                                   args.p_herald, args.horizon, args.seed)
                s = summarize(rows, args.deadline)
                summaries.append(s)
                reason = ' [GATE INFEASIBLE EVEN AT PERFECT INITIAL ALIGNMENT]' if s['infeasible_gate'] else ''
                print(f'  delta_limit={s["delta_threshold_ps"]:.3f}ps '
                      f'maxAdmission={s["max_possible_admission_p"]:.4f} '
                      f'finished={s["finished"]}/{s["n"]} '
                      f'by{args.deadline:g}us={s["deadline_pct"]:.1f}% '
                      f'mean(completed)={s["mean_us"]:.1f}us '
                      f'p95={s["p95_us"]:.1f}us '
                      f'measurements={s["measurements"]:.1f} '
                      f'calibrations={s["calibrations"]:.1f}{reason}',flush=True)
    with args.output.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f'\nSummary saved to {args.output}', flush=True)

if __name__ == '__main__':
    main()
