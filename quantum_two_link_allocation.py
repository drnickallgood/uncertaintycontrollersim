"""Two-link deadline-aware entanglement allocation experiment.

This is a controlled, HYPOTHETICAL architecture experiment, not a hardware
prediction. Two independent links have IDENTICAL physical configurations and
are sampled at independent action-boundary checkpoints after warm-up. Their
observable belief states and maintenance histories can differ. One request for
k ADMITTED heralds arrives and must be assigned to exactly one link.

Policies:
  rate         : identical nominal / long-run rate -> randomized fair tie.
  ready-first  : prefer a currently ready link; ties broken fairly.
  availability : estimate deadline probability using readiness / past control
                 history, but not continuous timing-belief statistics.
  belief       : same information PLUS estimated offset/variance/reliability.

Availability and belief predictors use exactly the same regularized logistic
model class and independently sampled training data, without true offset or
future outcomes. They use the pre-existing frozen adaptive v3 policy for
future sensing / calibration / attempts. For each held-out request, we simulate
both counterfactual link outcomes with RNG streams that are independent of
policy choices. This supports paired comparisons without granting any selector
future knowledge. Only the selected link counts toward the service result.

Evaluation reports held-out deadline-hit probability, choice disagreements,
paired policy differences with a bootstrap CI, and how many outcomes were
actually sensitive to which link was selected. The "hindsight either succeeds"
number is an UNATTAINABLE diagnostic ceiling, NOT an online baseline.

Times in microseconds; offsets/uncertainties in picoseconds; Q in ps^2/us.

Example:
  python -u quantum_two_link_allocation.py --train 100 --test 100 --bootstrap 200
  python -u quantum_two_link_allocation.py --train 1000 --test 1000 --bootstrap 1000

Requires NumPy plus quantum_workload_equal_start.py and
quantum_deadline_prediction.py in the same directory.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from quantum_workload_equal_start import Config
from quantum_deadline_prediction import (
    LogisticPredictor, brier, features, generate_data, request_outcome,
    warmup_checkpoint,
)


def train_estimators(cfg: Config, k: int, deadline: float, n: int,
                     seed: np.random.SeedSequence, warmup_min: float,
                     warmup_max: float) -> tuple[LogisticPredictor, LogisticPredictor, float]:
    xa, xb, y, _ = generate_data(
        cfg, k, deadline, n, seed, warmup_min, warmup_max
    )
    # Identical statistical model and training labels for the two stateful
    # predictors. Rate-only sees the marginal training rate, and is indifferent
    # between physically identical links at the same nominal/effective rate.
    return (LogisticPredictor().fit(xa, y),
            LogisticPredictor().fit(xb, y), float(y.mean()))


def make_test_requests(cfg: Config, k: int, deadline: float, n: int,
                       seed: np.random.SeedSequence, warmup_min: float,
                       warmup_max: float):
    """Return the paired request states and independently generated potentials.

    For each request two link states are drawn independently from the same
    warm-up distribution. Both *potential* outcomes are generated, regardless
    of which policy will choose either link. Each realization is fixed across
    all selectors, enabling a meaningful paired policy evaluation.
    """
    aa, bb, yy, readiness = [], [], [], []
    for child in seed.spawn(n):
        a_warm, b_warm, a_future, b_future = child.spawn(4)
        st_a, ctl_a = warmup_checkpoint(
            cfg, np.random.default_rng(a_warm), warmup_min, warmup_max)
        st_b, ctl_b = warmup_checkpoint(
            cfg, np.random.default_rng(b_warm), warmup_min, warmup_max)
        xa_a, xb_a, ready_a = features(st_a, ctl_a, cfg)
        xa_b, xb_b, ready_b = features(st_b, ctl_b, cfg)
        # The only difference between arms is the current stochastic link
        # state; neither is given priority by the generation code.
        y_a, _ = request_outcome(st_a, ctl_a, cfg, k, deadline,
                                 np.random.default_rng(a_future))
        y_b, _ = request_outcome(st_b, ctl_b, cfg, k, deadline,
                                 np.random.default_rng(b_future))
        aa.append([xa_a, xa_b])
        bb.append([xb_a, xb_b])
        yy.append([y_a, y_b])
        readiness.append([ready_a, ready_b])
    return (np.asarray(aa, dtype=float), np.asarray(bb, dtype=float),
            np.asarray(yy, dtype=int), np.asarray(readiness, dtype=float))


def paired_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
              bootstrap: int) -> tuple[float, float]:
    """Paired bootstrap CI of E[x-y]; preserves common held-out requests."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    estimates = np.empty(bootstrap)
    for j in range(bootstrap):
        estimates[j] = d[rng.integers(0, len(d), len(d))].mean()
    return tuple(float(v) for v in np.quantile(estimates, [.025, .975]))


def run_case(cfg: Config, k: int, deadline: float, train: int, test: int,
             bootstrap: int, seed: np.random.SeedSequence,
             warmup_min: float, warmup_max: float,
             raw: list[dict], summary: list[dict]):
    train_seed, test_seed, select_seed, boot_seed = seed.spawn(4)
    print(f'\nCase Q={cfg.q:g} ps^2/us, deadline={deadline:g}us, k={k}', flush=True)
    print(f'  Training on {train} independent single-link requests...', flush=True)
    avail_model, belief_model, train_hit = train_estimators(
        cfg, k, deadline, train, train_seed, warmup_min, warmup_max)
    print(f'  Evaluating {test} independent TWO-link requests...', flush=True)
    x_av, x_bl, yy, ready = make_test_requests(
        cfg, k, deadline, test, test_seed, warmup_min, warmup_max)
    p_av = avail_model.predict(x_av.reshape(2 * test, -1)).reshape(test, 2)
    p_bl = belief_model.predict(x_bl.reshape(2 * test, -1)).reshape(test, 2)
    selector_rng = np.random.default_rng(select_seed)
    # Nominal-rate ranking is a tie because both links are identical, so an
    # unbiased random tie-break is used rather than always privileging A.
    rate_choices = selector_rng.integers(0, 2, size=test)
    # A transparent, non-trained availability baseline: prefer currently
    # admitted links, else break ties at random. This prevents attributing
    # value to learned predictors just because a weak fitted model failed.
    ready_choices = np.where(ready[:,0] > ready[:,1], 0,
                            np.where(ready[:,1] > ready[:,0], 1,
                                     selector_rng.integers(0, 2, size=test)))
    avail_choices = np.argmax(p_av, axis=1)
    belief_choices = np.argmax(p_bl, axis=1)
    ix = np.arange(test)
    picks = {'rate': rate_choices, 'ready-first': ready_choices,
             'availability': avail_choices, 'belief': belief_choices}
    outcomes = {name: yy[ix, choose] for name, choose in picks.items()}
    print(f'  train marginal success={train_hit:.1%}; held-out per-link '
          f'success={yy.mean():.1%}; both-ready={np.mean(ready.sum(axis=1)==2):.1%}; '
          f'one-ready={np.mean(ready.sum(axis=1)==1):.1%}', flush=True)
    for name in ('rate', 'ready-first', 'availability', 'belief'):
        success = outcomes[name]
        print(f'  {name:12s} deadline hit={success.mean():.1%} '
              f'({int(success.sum())}/{test}); chooses B={picks[name].mean():.1%}',
              flush=True)
        summary.append(dict(q=cfg.q, deadline_us=deadline, k=k, n_train=train,
                            n_test=test, selector=name,
                            deadline_hit=float(success.mean()),
                            train_marginal_hit=train_hit,
                            heldout_arm_hit=float(yy.mean()),
                            availability_brier=brier(yy.ravel(),p_av.ravel()),
                            belief_brier=brier(yy.ravel(),p_bl.ravel()),
                            disagree_availability_belief=float(np.mean(avail_choices!=belief_choices)),
                            choice_sensitive=float(np.mean(yy[:,0]!=yy[:,1])),
                            hindsight_either=float(np.mean(yy.max(axis=1)))))
    print(f'  disagreement(availability,belief)='
          f'{np.mean(avail_choices!=belief_choices):.1%}; '
          f'choice-sensitive requests={np.mean(yy[:,0]!=yy[:,1]):.1%}',flush=True)
    for other in ('availability', 'ready-first', 'rate'):
        diff = float(np.mean(outcomes['belief'] - outcomes[other]))
        ci = paired_ci(outcomes['belief'],outcomes[other],
                       np.random.default_rng(boot_seed.spawn(1)[0]), bootstrap)
        wins = int(np.sum((outcomes['belief']==1)&(outcomes[other]==0)))
        losses = int(np.sum((outcomes['belief']==0)&(outcomes[other]==1)))
        print(f'  belief - {other}: {diff*100:+.2f} pp '
              f'95% paired CI [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp '
              f'(wins={wins}, losses={losses})',flush=True)
    print(f'  diagnostic hindsight either link succeeds: '
          f'{np.mean(yy.max(axis=1)):.1%} (not achievable with future-blind allocation)',
          flush=True)
    print(f'  predictive Brier per arm: availability='
          f'{brier(yy.ravel(),p_av.ravel()):.5f}, '
          f'belief={brier(yy.ravel(),p_bl.ravel()):.5f}', flush=True)
    for j in range(test):
        raw.append(dict(q=cfg.q, deadline_us=deadline, k=k, trial=j,
                        y_link_a=int(yy[j,0]),y_link_b=int(yy[j,1]),
                        ready_link_a=int(ready[j,0]),ready_link_b=int(ready[j,1]),
                        pred_available_a=float(p_av[j,0]),pred_available_b=float(p_av[j,1]),
                        pred_belief_a=float(p_bl[j,0]),pred_belief_b=float(p_bl[j,1]),
                        chosen_rate='AB'[rate_choices[j]],
                        chosen_ready_first='AB'[ready_choices[j]],
                        chosen_available='AB'[avail_choices[j]],
                        chosen_belief='AB'[belief_choices[j]],
                        hit_rate=int(outcomes['rate'][j]),
                        hit_ready_first=int(outcomes['ready-first'][j]),
                        hit_available=int(outcomes['availability'][j]),
                        hit_belief=int(outcomes['belief'][j])))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',type=int,default=1000)
    ap.add_argument('--test',type=int,default=1000)
    ap.add_argument('--bootstrap',type=int,default=500)
    ap.add_argument('--k',type=int,default=3)
    ap.add_argument('--cases',nargs='+',default=['0.09:150','0.40:1000'],
                    help='Q:deadline_us pairs, e.g. --cases 0.09:150 0.40:1000')
    ap.add_argument('--warmup-min',type=float,default=300.)
    ap.add_argument('--warmup-max',type=float,default=2000.)
    ap.add_argument('--seed',type=int,default=20261102)
    ap.add_argument('--output',type=Path,default=Path('two_link_allocation_summary.csv'))
    ap.add_argument('--raw-output',type=Path,default=Path('two_link_allocation_trials.csv'))
    args=ap.parse_args()
    if min(args.train,args.test,args.bootstrap,args.k)<1:
        ap.error('train/test/bootstrap/k must be positive')
    if not (0 <= args.warmup_min < args.warmup_max):
        ap.error('Require 0 <= warmup-min < warmup-max')
    seed=np.random.SeedSequence(args.seed)
    raw, summary=[],[]
    print('Two identical independent links, one request sent to ONE link;'
          ' idealized warm-up snapshots at action boundaries.',flush=True)
    print('Trained selectors see only observable input features;'
          ' held-out outcomes are fixed across selectors.',flush=True)
    print('Physical model and controller: toy Gaussian timing, frozen adaptive v3;'
          ' target is k ADMITTED HERALDS.',flush=True)
    for case, child in zip(args.cases,seed.spawn(len(args.cases))):
        try: q, deadline=[float(v) for v in case.split(':')]
        except ValueError: ap.error('Each case must be Q:deadline_us')
        if q<0 or deadline<=0: ap.error('Q must be nonnegative and deadlines positive')
        cfg=Config(q=q)
        cfg.validate()
        run_case(cfg,args.k,deadline,args.train,args.test,args.bootstrap,
                 child,args.warmup_min,args.warmup_max,raw,summary)
    for name,records in ((args.output,summary),(args.raw_output,raw)):
        name.parent.mkdir(parents=True,exist_ok=True)
        with name.open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(f'\nSaved summary: {args.output}',flush=True)
    print(f'Saved per-request paired outcomes: {args.raw_output}',flush=True)


if __name__=='__main__':
    main()
