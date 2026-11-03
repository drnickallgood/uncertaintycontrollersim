"""Source-scale and waveform robustness for timing-gated entanglement acquisition.

Requires quantum_workload_equal_start.py in the same folder.

Mode source: reported Ba+ lifetime 7850 ps; the Gaussian is a hypothetical
matched-standard-deviation alternative, not an experimentally measured pulse.
With Q/sensor/control times from earlier toy model, results are illustrative.

Mode stress: dimensionless stress Q*t_attempt/delta_limit_exponential**2;
uses hypothetical noise fractions and ideal p_herald; explicitly NOT experimental.

Mode rate: analytic acquisition lower-bound from observed remote-pair rate.
Not a Monte Carlo hardware prediction; no timing control simulation.
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import numpy as np
import quantum_workload_equal_start as sim
from quantum_wavepacket_robustness import timing_limit, fidelity


def test(shape, width, q, r, actuator, t_a, t_m, p_h, trials, pairs, deadline, horizon, seed):
    cfg=sim.Config(q=q, sigma=width, sensor_std=r, actuator_std=actuator,
                   t_attempt=t_a, t_measure=t_m, p_herald=p_h, horizon=horizon)
    cfg.validate()
    d=timing_limit(shape,width,cfg.min_fidelity)
    vcap=(d/2.5758293035489004)**2
    v_steady=(math.sqrt((q*t_m)**2+4*q*t_m*r*r)-q*t_m)/2
    def gate(st, c):
        variance=st.v+c.q*c.t_attempt
        if variance <= 0: return float(abs(st.m)<=d)
        s=math.sqrt(variance)
        return max(0.,min(1.,sim.normal_cdf((d-st.m)/s)-sim.normal_cdf((-d-st.m)/s)))
    old_gate,old_fid=sim.reliability_after_attempt,sim.fidelity
    sim.reliability_after_attempt=gate
    sim.fidelity=lambda delta,c: fidelity(shape,delta,width)
    try:
        max_admit=1 if q==0 else (2*sim.normal_cdf(d/math.sqrt(q*t_a))-1)
        impossible=max_admit+1e-12<cfg.min_reliability
        rows=[]
        for i,s in enumerate(np.random.SeedSequence(seed).spawn(trials)):
            if impossible:
                rows.append({'finished':0,'completion_us':math.nan,'measurements':0,'calibrations':0,'attempts':0,'all_pairs_meet_fidelity':0})
            else:
                rows.append(sim.run_trial(cfg,pairs,'adaptive',np.random.default_rng(s),'equal'))
    finally:
        sim.reliability_after_attempt,sim.fidelity=old_gate,old_fid
    complete=[x['completion_us'] for x in rows if x['finished']]
    hits=sum(x['finished'] and x['completion_us']<=deadline for x in rows)
    return dict(shape=shape,width_ps=width,q_ps2_per_us=q,sensor_ps=r,actuator_ps=actuator,
                attempt_us=t_a,p_herald=p_h,n=trials,delta_limit_ps=d,
                steady_state_variance_ps2=v_steady,centered_variance_budget_ps2=vcap,
                steady_state_condition=int(v_steady+q*t_a<=vcap),
                maximum_initial_gate_probability=max_admit,initial_gate_impossible=int(impossible),
                finished=len(complete),deadline_hits=hits,deadline_pct=100*hits/trials,
                mean_completion_us=np.mean(complete) if complete else math.nan,
                p95_completion_us=np.percentile(complete,95) if complete else math.nan,
                mean_measurements=np.mean([x['measurements'] for x in rows]),
                mean_calibrations=np.mean([x['calibrations'] for x in rows]),
                mean_attempts=np.mean([x['attempts'] for x in rows]),
                all_pairs_actual_good=sum(x['all_pairs_meet_fidelity'] for x in rows))


def poisson_tail_at_least_k(mean,k):
    # Poisson approximation to completed pairs at measured mean event rate.
    # Avoid catastrophic over/underflow for practical parameters.
    return max(0.,min(1.,1-math.exp(-mean)*sum(mean**j/math.factorial(j) for j in range(k)))) if mean<500 else 1.


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--mode',choices=['source','stress','rate'],default='source')
    a.add_argument('--trials',type=int,default=300)
    a.add_argument('--q',nargs='+',type=float,default=[0.09,0.4])
    a.add_argument('--normalized-q',nargs='+',type=float,default=[0.001,0.02,0.1,0.2])
    a.add_argument('--width-ps',type=float,default=7850.)
    a.add_argument('--sensor-ps',type=float,default=None)
    a.add_argument('--actuator-ps',type=float,default=None)
    a.add_argument('--attempt-us',type=float,default=10.)
    a.add_argument('--measure-us',type=float,default=5.)
    a.add_argument('--p-herald',type=float,default=0.25)
    a.add_argument('--pairs',type=int,default=3)
    a.add_argument('--deadline-us',type=float,default=1000.)
    a.add_argument('--horizon-us',type=float,default=5000.)
    a.add_argument('--measured-rate',type=float,default=0.35,help='Experimental accepted pairs per second')
    a.add_argument('--output',type=Path,default=Path('source_scale_validation.csv'))
    args=a.parse_args()
    if args.mode=='rate':
        rate=args.measured_rate
        if rate<=0: a.error('Rate must be positive')
        print('MEASURED RATE BENCHMARK (Saha et al. 2025 time-bin experiment)')
        print('Not our polarization protocol; no assumption that control reproduces experimental rate.')
        print(f'Rate={rate:g} accepted pairs/s; expected waiting time for {args.pairs} pairs = {args.pairs/rate:.3f} s')
        for sec in [0.001,0.1,1.,5.,10.,20.]:
            print(f'  P(at least {args.pairs} pairs by {sec:g} s) ~ {100*poisson_tail_at_least_k(rate*sec,args.pairs):.6f}% (Poisson approximation)')
        return
    if args.trials<1 or args.width_ps<=0 or args.pairs<1 or args.deadline_us>args.horizon_us: a.error('Invalid trials/width/pairs/horizon')
    if args.mode=='source':
        qs=args.q
        sensor=2. if args.sensor_ps is None else args.sensor_ps
        actuator=2. if args.actuator_ps is None else args.actuator_ps
        print('SOURCE-SCALE ENVELOPE COMPARISON: width=7.85 ns from Saha et al. time-bin source.')
        print('Gaussian same intensity stddev is HYPOTHETICAL; drift, sensor and control times remain TOY assumptions.')
    else:
        d=timing_limit('exponential',args.width_ps,.95)
        qs=[x*d*d/args.attempt_us for x in args.normalized_q]
        sensor=.05*args.width_ps if args.sensor_ps is None else args.sensor_ps
        actuator=.05*args.width_ps if args.actuator_ps is None else args.actuator_ps
        print('STRESS TEST: all drift and sensor settings HYPOTHETICAL; normalized Q is Q*t_attempt/delta_E^2.')
    if min(qs)<0 or sensor<=0 or actuator<0: a.error('Invalid noise configuration')
    result=[]
    for q in qs:
        for shape in ('gaussian','exponential'):
            print(f'Running {shape:11s} Q={q:.6g} width={args.width_ps:g}ps pH={args.p_herald:g} N={args.trials}',flush=True)
            row=test(shape,args.width_ps,q,sensor,actuator,args.attempt_us,args.measure_us,args.p_herald,args.trials,args.pairs,args.deadline_us,args.horizon_us,20261102)
            result.append(row)
            print(f'  admissibility max={row["maximum_initial_gate_probability"]:.5f} steadyOK={bool(row["steady_state_condition"])} '
                  f'completed={row["finished"]}/{args.trials} deadline={row["deadline_pct"]:.1f}% '
                  f'mean={row["mean_completion_us"]:.1f}us measurements={row["mean_measurements"]:.1f}',flush=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0].keys()))
        w.writeheader(); w.writerows(result)
    print(f'Summary saved to {args.output}')

if __name__=='__main__': main()
