"""Compare aggregate, availability-aware, and timing-belief deadline predictors.

Reuse the *frozen adaptive* controller primitives in quantum_workload_equal_start.py.
The simulated service request is for k heralded Bell pairs by a specified deadline.
Pair quality remains latent: only admitted HERALDS count, as in the original model.

Experimental design:
  * Each request arrives after an independently simulated continuous-link warm-up.
  * Request states are taken at an action-completion boundary (not mid-operation).
  * At request arrival, hidden timing offset is sampled from N(m,v); neither
    architecture predictor ever receives the hidden offset.
  * Training and held-out request states are independent. Every predictor sees
    exactly the same held-out requests and deadlines.
  * All predictors are fit with the same regularized logistic model. The
    aggregate comparator uses only an intercept (calibrated marginal base rate).
  * Availability-aware gets the binary admission status plus operational history,
    but not m, v, or their continuously valued reliability prediction.
  * Belief-conditioned gets those same features PLUS timing-belief information.

These are capacity-limited REFERENCE predictors, not state-of-the-art quantum
network schedulers. An improvement in predictive skill is not by itself an
architecture scheduling speedup or novel quantum link-layer protocol.

Units: offset/std/width in ps; time in microseconds; Q in ps^2/us.

Examples:
  python -u quantum_deadline_prediction.py --train 200 --test 200 --bootstrap 200
  python -u quantum_deadline_prediction.py --train 2000 --test 2000 --bootstrap 1000
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from quantum_workload_equal_start import (
    Config, State, advance, attempt, calibrate, measure,
    reliability_after_attempt, should_calibrate,
)


@dataclass
class Controller:
    recovery_streak: int = 0
    force_measure: bool = False
    last_action: str = 'none'
    last_measure_t: float | None = None
    last_calibration_t: float | None = None


def next_action(st: State, ctl: Controller, cfg: Config) -> str:
    if ctl.force_measure:
        return 'measure'
    if reliability_after_attempt(st, cfg) >= cfg.min_reliability:
        return 'attempt'
    if ctl.recovery_streak >= cfg.maintenance_limit:
        return 'idle'
    return 'calibrate' if should_calibrate(st) else 'measure'


def execute_action(st: State, ctl: Controller, cfg: Config,
                   rng: np.random.Generator, action: str) -> None:
    """Exactly follow the adaptive-v3 state transitions in the existing code."""
    if action == 'measure':
        measure(st, cfg, rng)
        ctl.force_measure = False
        ctl.recovery_streak += 1
        ctl.last_measure_t = st.t
    elif action == 'calibrate':
        calibrate(st, cfg, rng)
        ctl.force_measure = True
        ctl.recovery_streak += 1
        ctl.last_calibration_t = st.t
    elif action == 'attempt':
        attempt(st, cfg, rng)
        ctl.recovery_streak = 0
    elif action == 'idle':
        advance(st, cfg.idle_duration, cfg, rng)
        st.idle_time += cfg.idle_duration
        ctl.recovery_streak = 0
    else:
        raise ValueError(action)
    ctl.last_action = action


def action_duration(cfg: Config, action: str) -> float:
    return {
        'attempt': cfg.t_attempt,
        'measure': cfg.t_measure,
        'calibrate': cfg.t_calibrate,
        'idle': cfg.idle_duration,
    }[action]


def warmup_checkpoint(cfg: Config, rng: np.random.Generator,
                      minimum: float, maximum: float) -> tuple[State, Controller]:
    """Sample a link state after continued operation, at an action boundary."""
    target = rng.uniform(minimum, maximum)
    st, ctl = State(), Controller()
    while st.t < target:
        action = next_action(st, ctl, cfg)
        execute_action(st, ctl, cfg, rng, action)
    return st, ctl


def request_outcome(st0: State, ctl0: Controller, cfg: Config, k: int,
                    deadline_us: float, rng: np.random.Generator) -> tuple[int, float]:
    """Roll forward from posterior-sampled latent state to a fixed deadline."""
    # At a request, use the posterior N(m,v), not the hidden warm-up value.
    # This ensures request labels genuinely depend on the information available
    # to the controller and does not leak an unobservable physical state.
    initial_delta = (float(rng.normal(st0.m, math.sqrt(st0.v)))
                     if st0.v > 0 else float(st0.m))
    st = State(delta=initial_delta, m=st0.m, v=st0.v)
    ctl = replace(ctl0, last_measure_t=None, last_calibration_t=None)
    # The last-action ages are *features* of the warm-up state. They are not
    # consulted by the frozen adaptive controller during the request rollout.
    while st.t < deadline_us and st.heralds < k:
        action = next_action(st, ctl, cfg)
        if action_duration(cfg, action) > deadline_us - st.t + 1e-10:
            break
        execute_action(st, ctl, cfg, rng, action)
    return int(st.heralds >= k), st.t


def features(st: State, ctl: Controller, cfg: Config) -> tuple[np.ndarray, np.ndarray, float]:
    ready = float(reliability_after_attempt(st, cfg) >= cfg.min_reliability)
    # All feature values below are observable at request arrival. In
    # particular, availability-aware receives READY but no continuous belief.
    since_measure = st.t if ctl.last_measure_t is None else st.t - ctl.last_measure_t
    since_cal = st.t if ctl.last_calibration_t is None else st.t - ctl.last_calibration_t
    maintenance_time = (st.measurements * cfg.t_measure +
                        st.calibrations * cfg.t_calibrate + st.idle_time)
    duty = maintenance_time / max(1., st.t)
    ava = np.array([
        ready,
        float(ctl.force_measure),
        ctl.recovery_streak / max(1, cfg.maintenance_limit),
        duty,
        min(since_measure, 250.) / 250.,
        min(since_cal, 500.) / 500.,
        float(ctl.last_action == 'measure'),
        float(ctl.last_action == 'calibrate'),
        float(ctl.last_action == 'idle'),
    ], dtype=float)
    mu = st.m / cfg.sigma
    var = st.v / (cfg.sigma ** 2)
    pvalid = reliability_after_attempt(st, cfg)
    # Equal model class, richer observable representation.
    extra = np.array([
        mu, abs(mu), mu * mu,
        math.sqrt(max(0., var)), var,
        pvalid, pvalid * pvalid,
        max(0., cfg.q * cfg.t_attempt + st.v) / (cfg.sigma ** 2),
    ], dtype=float)
    return ava, np.concatenate([ava, extra]), ready


def generate_data(cfg: Config, k: int, deadline: float, count: int,
                  seed: np.random.SeedSequence, warmup_min: float,
                  warmup_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ava_rows, belief_rows, labels, readiness = [], [], [], []
    for child in seed.spawn(count):
        warm_seed, outcome_seed = child.spawn(2)
        st, ctl = warmup_checkpoint(cfg, np.random.default_rng(warm_seed),
                                    warmup_min, warmup_max)
        a, b, ready = features(st, ctl, cfg)
        y, _ = request_outcome(st, ctl, cfg, k, deadline,
                               np.random.default_rng(outcome_seed))
        ava_rows.append(a)
        belief_rows.append(b)
        labels.append(y)
        readiness.append(ready)
    return (np.asarray(ava_rows), np.asarray(belief_rows),
            np.asarray(labels, dtype=float), np.asarray(readiness))


class LogisticPredictor:
    """Numpy-only L2-regularized logistic regression with standardized features."""
    def __init__(self, ridge: float = 3.0):
        self.ridge = ridge

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'LogisticPredictor':
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.center = x.mean(axis=0) if x.shape[1] else np.zeros(0)
        self.scale = x.std(axis=0) if x.shape[1] else np.ones(0)
        self.scale = np.where(self.scale > 1e-9, self.scale, 1.)
        xs = (x - self.center) / self.scale
        X = np.column_stack([np.ones(len(xs)), xs])
        ybar = np.clip((sum(y) + .5) / (len(y) + 1.), 1e-5, 1 - 1e-5)
        weights = np.zeros(X.shape[1]); weights[0] = math.log(ybar / (1-ybar))
        regularize = np.diag([0.] + [self.ridge] * (X.shape[1]-1))
        for _ in range(70):
            eta = np.clip(X @ weights, -30., 30.)
            prob = 1. / (1. + np.exp(-eta))
            grad = X.T @ (prob - y) + regularize @ weights
            diag = np.maximum(prob * (1. - prob), 1e-7)
            hess = X.T @ (X * diag[:, None]) + regularize
            try:
                change = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError('Logistic fit failed') from exc
            # Safeguarded line search for near-separable training data.
            old_obj = np.logaddexp(0, eta).sum() - y @ eta + .5 * weights @ regularize @ weights
            step = 1.
            while step > 1e-6:
                candidate = weights - step * change
                neweta = X @ candidate
                new_obj = (np.logaddexp(0, neweta).sum() - y @ neweta +
                           .5 * candidate @ regularize @ candidate)
                if new_obj <= old_obj + 1e-8:
                    break
                step *= .5
            weights = weights - step * change
            if np.max(np.abs(step * change)) < 1e-7:
                break
        self.weights = weights
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        X = np.column_stack([np.ones(len(x)), (x - self.center) / self.scale])
        return np.clip(1. / (1. + np.exp(-np.clip(X @ self.weights, -30., 30.))),
                       1e-6, 1-1e-6)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p-y)**2))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(y * np.log(p) + (1-y) * np.log1p(-p)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins+1)
    ix = np.minimum(bins-1, np.searchsorted(edges[1:], p, side='right'))
    return float(sum((ix == j).mean() * abs(y[ix == j].mean() - p[ix == j].mean())
                     for j in range(bins) if np.any(ix == j)))


def paired_ci_brier(y: np.ndarray, a: np.ndarray, b: np.ndarray,
                     rng: np.random.Generator, count: int) -> tuple[float, float]:
    """CI for BS(a)-BS(b), positive => b is better."""
    diffs = (a-y)**2 - (b-y)**2
    sample = [np.mean(diffs[rng.integers(0, len(diffs), size=len(diffs))])
              for _ in range(count)]
    return float(np.quantile(sample, .025)), float(np.quantile(sample, .975))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=int, default=1000, help='Independent training request states per case')
    ap.add_argument('--test', type=int, default=1000, help='Independent held-out request states per case')
    ap.add_argument('--bootstrap', type=int, default=500, help='Paired bootstrap replicates')
    ap.add_argument('--k', type=int, default=3, help='Heralded Bell pairs requested')
    ap.add_argument('--cases', nargs='+', default=['0.09:150', '0.40:1000'],
                    help='Q:deadline_us, for example --cases 0.09:150 0.40:1000')
    ap.add_argument('--warmup-min', type=float, default=300., help='Initial min warm-up us')
    ap.add_argument('--warmup-max', type=float, default=2000., help='Initial max warm-up us')
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--output', type=Path, default=Path('deadline_prediction_summary.csv'))
    ap.add_argument('--raw-output', type=Path, default=Path('deadline_prediction_heldout.csv'))
    args = ap.parse_args()
    if min(args.train, args.test, args.bootstrap, args.k) < 1 or not (0 <= args.warmup_min < args.warmup_max):
        ap.error('Use positive sample sizes and 0 <= warmup-min < warmup-max')
    root = np.random.SeedSequence(args.seed)
    rows, raw = [], []
    print('Model: frozen adaptive v3; independent realistic-policy warm-up states.', flush=True)
    print('Target: k ADMITTED HERALDS by deadline; not verified successful gates.', flush=True)
    print('Models: calibrated marginal; readiness/history; readiness/history + timing belief.', flush=True)
    for case, parent_seed in zip(args.cases, root.spawn(len(args.cases))):
        try:
            q, deadline = (float(v) for v in case.split(':'))
        except ValueError:
            ap.error('Each --cases entry must be Q:deadline_us')
        cfg = Config(q=q)
        cfg.validate()
        if deadline <= 0:
            ap.error('Deadline must be positive')
        training_seed, testing_seed, boot_seed = parent_seed.spawn(3)
        print(f'\nCase Q={q:g} ps^2/us deadline={deadline:g}us k={args.k}', flush=True)
        print(f'  Training {args.train} independent request states...', flush=True)
        xa_tr, xb_tr, y_tr, ready_tr = generate_data(
            cfg, args.k, deadline, args.train, training_seed, args.warmup_min, args.warmup_max)
        print(f'  Testing  {args.test} HELD-OUT request states...', flush=True)
        xa_te, xb_te, y_te, ready_te = generate_data(
            cfg, args.k, deadline, args.test, testing_seed, args.warmup_min, args.warmup_max)
        x0tr, x0te = np.empty((len(y_tr), 0)), np.empty((len(y_te), 0))
        fitted = {
            'aggregate': LogisticPredictor().fit(x0tr, y_tr).predict(x0te),
            'availability': LogisticPredictor().fit(xa_tr, y_tr).predict(xa_te),
            'belief': LogisticPredictor().fit(xb_tr, y_tr).predict(xb_te),
        }
        print(f'  train hit={y_tr.mean():.1%} held-out hit={y_te.mean():.1%} '
              f'ready-at-request={ready_te.mean():.1%}', flush=True)
        for name, p in fitted.items():
            score = brier(y_te, p)
            row = dict(q=q, deadline_us=deadline, k=args.k, model=name,
                       n_train=args.train, n_test=args.test,
                       train_hit_rate=float(y_tr.mean()),
                       heldout_hit_rate=float(y_te.mean()),
                       mean_pred=float(p.mean()), brier=score,
                       log_loss=logloss(y_te, p), ece10=ece(y_te,p),
                       ready_at_request=float(ready_te.mean()))
            rows.append(row)
            print(f'  {name:12s} Brier={score:.5f} logloss={row["log_loss"]:.5f} '
                  f'ECE={row["ece10"]:.4f} mean_pred={row["mean_pred"]:.1%}', flush=True)
        ci = paired_ci_brier(y_te, fitted['availability'], fitted['belief'],
                             np.random.default_rng(boot_seed), args.bootstrap)
        gap = brier(y_te, fitted['availability']) - brier(y_te, fitted['belief'])
        print(f'  Brier(availability) - Brier(belief) = {gap:+.5f} '
              f'95% paired bootstrap CI [{ci[0]:+.5f}, {ci[1]:+.5f}]', flush=True)
        for i in range(len(y_te)):
            raw.append(dict(q=q, deadline_us=deadline, k=args.k, trial=i,
                            success_by_deadline=int(y_te[i]), ready_at_request=int(ready_te[i]),
                            pred_aggregate=float(fitted['aggregate'][i]),
                            pred_availability=float(fitted['availability'][i]),
                            pred_belief=float(fitted['belief'][i])))
    for path, data in [(args.output,rows),(args.raw_output,raw)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(f'\nSaved summary: {args.output}',flush=True)
    print(f'Saved held-out predictions: {args.raw_output}',flush=True)


if __name__ == '__main__':
    main()
