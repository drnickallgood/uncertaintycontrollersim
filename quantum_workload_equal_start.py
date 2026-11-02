"""Workload-driven Monte Carlo for a timing-limited heralded quantum link.

Run: python quantum_workload_equal_start.py --trials 1000 --pairs 1 2 3 --q 0.09

Model times: microseconds; timing offsets and stddevs: picoseconds;
Q: ps^2 / microsecond.

IMPORTANT: Pair fidelity is latent. An application consumes a herald admitted by
its predictive reliability gate; the simulator observes actual fidelity only to
report diagnostic quality, not to make scheduling or completion decisions.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    q: float = 0.09
    sigma: float = 10.0
    min_fidelity: float = 0.95
    min_reliability: float = 0.99
    sensor_std: float = 2.0
    actuator_std: float = 2.0
    t_measure: float = 5.0
    t_calibrate: float = 15.0
    t_attempt: float = 10.0
    p_herald: float = 0.25
    horizon: float = 100_000.0
    reactive_period: int = 100
    maintenance_limit: int = 8
    idle_duration: float = 10.0

    def validate(self) -> None:
        if not (0 < self.min_fidelity < 1):
            raise ValueError('min_fidelity must lie in (0,1)')
        if not (0 < self.min_reliability < 1):
            raise ValueError('min_reliability must lie in (0,1)')
        if self.q < 0 or self.sensor_std <= 0 or self.actuator_std < 0:
            raise ValueError('Invalid noise configuration')
        if min(self.t_measure, self.t_calibrate, self.t_attempt,
               self.idle_duration, self.horizon) <= 0:
            raise ValueError('All durations must be positive')
        if not (0 <= self.p_herald <= 1):
            raise ValueError('p_herald must lie in [0,1]')
        if self.reactive_period < 1 or self.maintenance_limit < 1:
            raise ValueError('Counts must be positive')


@dataclass
class State:
    t: float = 0.0
    delta: float = 0.0
    m: float = 0.0
    v: float = 0.0
    attempts: int = 0
    heralds: int = 0
    actual_good: int = 0
    measurements: int = 0
    calibrations: int = 0
    idle_time: float = 0.0
    worst_pair_fidelity: float = 1.0


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def fidelity(delta: float, cfg: Config) -> float:
    return 0.5 * (1.0 + math.exp(-delta * delta / (4.0 * cfg.sigma**2)))


def reliability_after_attempt(st: State, cfg: Config) -> float:
    """Predicted chance of meeting fidelity threshold at END of attempt."""
    if cfg.min_fidelity <= 0.5:
        return 1.0
    delta_max = 2.0 * cfg.sigma * math.sqrt(-math.log(2.0 * cfg.min_fidelity - 1.0))
    pred_var = st.v + cfg.q * cfg.t_attempt
    if pred_var <= 0:
        return float(abs(st.m) <= delta_max)
    sd = math.sqrt(pred_var)
    return max(0.0, min(1.0,
        normal_cdf((delta_max - st.m) / sd) -
        normal_cdf((-delta_max - st.m) / sd)))


def advance(st: State, dt: float, cfg: Config, rng: np.random.Generator) -> None:
    st.delta += rng.normal(0.0, math.sqrt(cfg.q * dt)) if cfg.q else 0.0
    st.v += cfg.q * dt
    st.t += dt


def measure(st: State, cfg: Config, rng: np.random.Generator) -> None:
    advance(st, cfg.t_measure, cfg, rng)
    y = st.delta + rng.normal(0.0, cfg.sensor_std)
    gain = st.v / (st.v + cfg.sensor_std**2)
    st.m += gain * (y - st.m)
    st.v = max(0.0, (1.0 - gain) * st.v)
    st.measurements += 1


def calibrate(st: State, cfg: Config, rng: np.random.Generator) -> None:
    advance(st, cfg.t_calibrate, cfg, rng)
    correction = st.m
    st.delta = st.delta - correction + rng.normal(0.0, cfg.actuator_std)
    st.m = 0.0
    st.v += cfg.actuator_std**2
    st.calibrations += 1


def attempt(st: State, cfg: Config, rng: np.random.Generator) -> bool:
    """Return True if a herald was observed, not if latent fidelity was high."""
    advance(st, cfg.t_attempt, cfg, rng)
    st.attempts += 1
    if rng.random() >= cfg.p_herald:
        return False
    st.heralds += 1
    f = fidelity(st.delta, cfg)
    st.worst_pair_fidelity = min(st.worst_pair_fidelity, f)
    st.actual_good += int(f >= cfg.min_fidelity)
    return True


def should_calibrate(st: State) -> bool:
    return abs(st.m) >= 3.0 and abs(st.m) >= 1.5 * math.sqrt(st.v)


def run_trial(cfg: Config, k: int, policy: str, rng: np.random.Generator,
              start_mode: str = "equal") -> dict:
    st = State()
    recovery_streak = 0
    force_measure = False
    # For reactive periodic: scheduled measure, calibrate, measure, N attempts.
    schedule = ('measure', 'calibrate', 'measure') + ('attempt',) * cfg.reactive_period
    if start_mode not in ('equal', 'periodic'):
        raise ValueError('start_mode must be equal or periodic')
    # Under equal start, both policies may attempt immediately from the same
    # (delta, m, v)=(0,0,0) state, provided the reliability gate permits it.
    # After the first reactive attempt, periodic maintenance remains scheduled
    # every N attempts. Original compulsory-start behavior remains available.
    scheduled_index = 3 if start_mode == 'equal' else 0
    # A pending attempt remains pending when reactive recovery interrupts it.
    # For the periodic schedule, the next index is changed ONLY when an
    # actually scheduled action executes.
    while st.t < cfg.horizon and st.heralds < k:
        remaining = cfg.horizon - st.t
        if force_measure:
            action = 'measure'
            scheduled = False
        elif policy == 'adaptive':
            if reliability_after_attempt(st, cfg) >= cfg.min_reliability:
                action = 'attempt'
            elif recovery_streak >= cfg.maintenance_limit:
                action = 'idle'
            else:
                action = 'calibrate' if should_calibrate(st) else 'measure'
            scheduled = False
        elif policy == 'reactive':
            scheduled_action = schedule[scheduled_index]
            if scheduled_action != 'attempt':
                action = scheduled_action
                scheduled = True
            elif reliability_after_attempt(st, cfg) >= cfg.min_reliability:
                action = 'attempt'
                scheduled = True
            elif recovery_streak >= cfg.maintenance_limit:
                action = 'idle'
                scheduled = False
            else:
                action = 'calibrate' if should_calibrate(st) else 'measure'
                scheduled = False
        else:
            raise ValueError('Unknown policy: ' + policy)
        dt = {'measure': cfg.t_measure, 'calibrate': cfg.t_calibrate,
              'attempt': cfg.t_attempt, 'idle': cfg.idle_duration}[action]
        if dt > remaining + 1e-10:
            break  # Do not execute partial physical operations.
        if action == 'measure':
            measure(st, cfg, rng)
            force_measure = False
            recovery_streak += 1
        elif action == 'calibrate':
            calibrate(st, cfg, rng)
            # The periodic schedule already follows calibration with measurement.
            force_measure = not scheduled
            recovery_streak += 1
        elif action == 'attempt':
            # The reliability check was performed before advancing the attempt.
            attempt(st, cfg, rng)
            recovery_streak = 0
        else:
            advance(st, cfg.idle_duration, cfg, rng)
            st.idle_time += cfg.idle_duration
            recovery_streak = 0
            # In reactive mode, the pending rejected attempt slot is skipped
            # after a prolonged failed recovery; adaptive mode simply retries.
            if policy == 'reactive':
                scheduled_index = (scheduled_index + 1) % len(schedule)
        if scheduled:
            scheduled_index = (scheduled_index + 1) % len(schedule)
    finished = st.heralds >= k
    return {
        'policy': policy, 'start_mode': start_mode, 'pairs_required': k, 'q': cfg.q,
        'finished': int(finished),
        'completion_us': st.t if finished else math.nan,
        'observed_us': st.t,
        'heralded_pairs': st.heralds,
        'all_pairs_meet_fidelity': int(finished and st.actual_good == k),
        'actual_good_pairs': st.actual_good,
        'worst_fidelity': st.worst_pair_fidelity if st.heralds else math.nan,
        'attempts': st.attempts,
        'measurements': st.measurements,
        'calibrations': st.calibrations,
        'idle_us': st.idle_time,
    }


def report(rows: list[dict], cfg: Config) -> None:
    for key in sorted(set((r['policy'], r['pairs_required'], r['q']) for r in rows)):
        group = [r for r in rows if (r['policy'], r['pairs_required'], r['q']) == key]
        done = np.array([r['completion_us'] for r in group if r['finished']], dtype=float)
        n = len(group)
        print(f'\n{key[0]:8s} k={key[1]} Q={key[2]:.3f} N={n}')
        print(f'  completed: {len(done)}/{n} ({len(done)/n:.1%}); '
              f'all k actual-good: {sum(r["all_pairs_meet_fidelity"] for r in group)}/{n} '
              '(diagnostic only)')
        if done.size:
            print('  conditional-on-completion latency [us]: '
                  f'mean={np.mean(done):.1f} p50={np.percentile(done,50):.1f} '
                  f'p95={np.percentile(done,95):.1f} p99={np.percentile(done,99):.1f}')
        if len(done) < n:
            print(f'  CENSORED: {n-len(done)} trials did not complete by '
                  f'{cfg.horizon:g} us; percentiles above EXCLUDE censored trials.')
        print(f'  mean attempts={np.mean([r["attempts"] for r in group]):.1f}; '
              f'measures={np.mean([r["measurements"] for r in group]):.1f}; '
              f'calibrations={np.mean([r["calibrations"] for r in group]):.1f}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trials', type=int, default=1000)
    ap.add_argument('--start-mode', choices=['equal', 'periodic'], default='equal',
                    help='equal removes compulsory startup maintenance for reactive')
    ap.add_argument('--pairs', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--q', type=float, nargs='+', default=[0.09])
    ap.add_argument('--p-herald', type=float, default=0.25)
    ap.add_argument('--horizon', type=float, default=100_000.0)
    ap.add_argument('--seed', type=int, default=20261101)
    ap.add_argument('--output', type=Path, default=Path('workload_trials.csv'))
    args = ap.parse_args()
    if args.trials < 1 or not args.pairs or min(args.pairs) < 1:
        ap.error('trials and all pair counts must be positive')
    master = np.random.SeedSequence(args.seed)
    settings = [(q, k, policy) for q in args.q for k in args.pairs
                for policy in ('adaptive', 'reactive')]
    child_seeds = master.spawn(len(settings))
    rows = []
    for (q, k, policy), child_seed in zip(settings, child_seeds):
        print(f'Running {policy} k={k} Q={q:.3f} start={args.start_mode} N={args.trials}', flush=True)
        cfg = Config(q=q, p_herald=args.p_herald, horizon=args.horizon)
        cfg.validate()
        trial_seeds = child_seed.spawn(args.trials)
        for t, seed in enumerate(trial_seeds):
            result = run_trial(cfg, k, policy, np.random.default_rng(seed), args.start_mode)
            result['trial'] = t
            rows.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for q in args.q:
        cfg = Config(q=q, p_herald=args.p_herald, horizon=args.horizon)
        report([r for r in rows if r['q'] == q], cfg)
    print(f'\nStart mode: {args.start_mode}')
    print(f'Raw trials saved to {args.output}')


if __name__ == '__main__':
    main()
