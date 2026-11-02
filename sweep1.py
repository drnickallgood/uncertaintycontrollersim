
import numpy as np
from math import erf, sqrt


attempts = 0
heralds = 0
useful_pairs = 0


def advance_state(delta, m, v, Q, dt, rng):
    w = rng.normal(0, np.sqrt(Q * dt))
    return delta + w, m, v + Q * dt


def measure_and_update(delta, m, v, r, rng):
    y = delta + rng.normal(0, r)
    R = r**2
    K = v / (v + R)
    return y, m + K * (y - m), (1 - K) * v


def calibrate(delta, m, v, Q, dt, a, rng):
    delta, m, v = advance_state(delta, m, v, Q, dt, rng)
    u = m
    eta = rng.normal(0, a)
    return delta - u + eta, m - u, v + a**2


def fidelity(delta, sigma):
    return 0.5 * (1 + np.exp(-delta**2 / (4 * sigma**2)))


def execute_attempt(delta, m, v, Q, dt,
                    sigma, p_h, F_min, rng):
    delta, m, v = advance_state(delta, m, v, Q, dt, rng)

    heralded = rng.random() < p_h
    F = fidelity(delta, sigma) if heralded else None
    useful = heralded and F >= F_min

    return delta, m, v, heralded, F, useful

# SChedudling

def run_fixed_baseline(seed=42, n_attempts=8):
    rng = np.random.default_rng(seed)

    # Simulation parameters
    T_end = 1000.0
    Q = 0.09
    r = 2.0
    a = 2.0
    sigma = 10.0
    p_h = 0.25
    F_min = 0.95
    fidelity_failures = 0
    predicted_violations = 0
    reliability_sum = 0.0
    measurements = 0
    calibrations = 0

    # Initial state
    t = 0.0
    delta = 5.0
    m = 4.0
    v = 4.0

    attempts = 0
    heralds = 0
    useful_pairs = 0

    #schedule = ["measure", "calibrate"] + ["attempt"] * 8
    #schedule = ["measure", "calibrate", "measure"] + ["attempt"] * 8
    schedule = ["measure", "calibrate", "measure"] + (["attempt"] * n_attempts)


    action_index = 0

    while t < T_end:

        action = schedule[action_index]

        dt = {
            "measure": 5.0,
            "calibrate": 15.0,
            "attempt": 10.0
        }[action]

        # Never execute an action past the deadline
        if t + dt > T_end:
            break

        if action == "measure":
            delta, m, v = advance_state(
                delta, m, v, Q, dt, rng
            )
            y, m, v = measure_and_update(
                delta, m, v, r, rng
            )
            measurements += 1

        elif action == "calibrate":
            delta, m, v = calibrate(
                delta, m, v, Q, dt, a, rng
            )
            calibrations += 1

        elif action == "attempt":
            reliability = predicted_reliability(
                m, v, Q, dt, sigma, F_min
            )

            reliability_sum += reliability

            if reliability < 0.99:
                predicted_violations += 1

            delta, m, v, heralded, F, useful = execute_attempt(
                delta, m, v, Q, dt,
                sigma, p_h, F_min, rng
            )

            attempts += 1
            heralds += int(heralded)
            useful_pairs += int(useful)

            if heralded and not useful:
                fidelity_failures += 1

        # Advance time exactly once per action
        t += dt

        # Move to next action, wrapping each cycle
        action_index = (action_index + 1) % len(schedule)

        if t >= T_end:
            break

    throughput = useful_pairs / (T_end * 1e-6)
    assert t <= T_end + 1e-9
    assert heralds == useful_pairs + fidelity_failures

    return {
        "attempts": attempts,
        "heralds": heralds,
        "useful_pairs": useful_pairs,
        "throughput_pairs_per_sec": throughput,
        "fidelity_failures": fidelity_failures,
        "predicted_violations": predicted_violations,
        "mean_predicted_reliability": (
            reliability_sum / attempts if attempts else 0.0),
        "measurements": measurements,
        "calibrations": calibrations,
        "elapsed_us": t
    }


def run_gated_fixed(seed=42, n_attempts=8,
                    reliability_target=0.99):

    rng = np.random.default_rng(seed)

    # Same physical parameters as other controllers
    T_end = 1000.0
    Q = 0.09
    r = 2.0
    a = 2.0
    sigma = 10.0
    p_h = 0.25
    F_min = 0.95

    # Initial state
    t = 0.0
    delta = 5.0
    m = 4.0
    v = 4.0

    # Counters
    attempts = 0
    heralds = 0
    useful_pairs = 0
    fidelity_failures = 0

    measurements = 0
    calibrations = 0
    skipped_attempts = 0
    constraint_violations = 0
    reliability_sum = 0.0

    # Fixed maintenance schedule
    schedule = (
        ["measure", "calibrate", "measure"]
        + ["attempt"] * n_attempts
    )

    action_index = 0

    while t < T_end:

        action = schedule[action_index]

        dt = {
            "measure": 5.0,
            "calibrate": 15.0,
            "attempt": 10.0
        }[action]

        if t + dt > T_end:
            break

        if action == "measure":

            delta, m, v = advance_state(
                delta, m, v, Q, dt, rng
            )

            y, m, v = measure_and_update(
                delta, m, v, r, rng
            )

            measurements += 1

        elif action == "calibrate":

            delta, m, v = calibrate(
                delta, m, v, Q, dt, a, rng
            )

            calibrations += 1

        elif action == "attempt":

            reliability = predicted_reliability(
                m, v, Q, dt, sigma, F_min
            )

            if reliability >= reliability_target:

                (
                    delta, m, v,
                    heralded, F, useful
                ) = execute_attempt(
                    delta, m, v, Q, dt,
                    sigma, p_h, F_min, rng
                )

                attempts += 1
                heralds += int(heralded)
                useful_pairs += int(useful)
                reliability_sum += reliability

                if heralded and not useful:
                    fidelity_failures += 1

                if reliability < reliability_target:
                    constraint_violations += 1

            else:
                # Slot is skipped, but the physical link
                # still drifts and the clock advances.
                delta, m, v = advance_state(
                    delta, m, v, Q, dt, rng
                )

                skipped_attempts += 1

        t += dt
        action_index = (action_index + 1) % len(schedule)

    throughput = useful_pairs / (T_end * 1e-6)

    assert t <= T_end + 1e-9
    assert heralds == useful_pairs + fidelity_failures

    return {
        "attempts": attempts,
        "heralds": heralds,
        "useful_pairs": useful_pairs,
        "fidelity_failures": fidelity_failures,
        "measurements": measurements,
        "calibrations": calibrations,
        "skipped_attempts": skipped_attempts,
        "constraint_violations": constraint_violations,
        "elapsed_us": t,
        "throughput_pairs_per_sec": throughput,
        "mean_predicted_reliability": (
            reliability_sum / attempts
            if attempts else float("nan")
        )
    }




def normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def predicted_reliability(m, v, Q, dt, sigma, F_min):
    # Predict variance at attempt completion
    v_future = v + Q * dt

    # Timing window corresponding to F >= F_min
    d_max = 2 * sigma * np.sqrt(-np.log(2 * F_min - 1))

    if v_future <= 0:
        return float(abs(m) <= d_max)

    std = np.sqrt(v_future)

    upper = (d_max - m) / std
    lower = (-d_max - m) / std

    return normal_cdf(upper) - normal_cdf(lower)




#def run_adaptive(seed=42, reliability_target=0.99):
def run_adaptive(seed=42, reliability_target=0.99, Q=0.09, r=2.0):
    rng = np.random.default_rng(seed)

    # Physical parameters: unchanged
    T_end = 1000.0
    #Q = 0.09
    #r = 2.0
    a = 2.0
    sigma = 10.0
    p_h = 0.25
    F_min = 0.95

    # Action durations (microseconds)
    measure_dt = 5.0
    calibrate_dt = 15.0
    attempt_dt = 10.0
    idle_dt = 10.0

    # Initial state
    t = 0.0
    delta = 5.0
    m = 4.0
    v = 4.0

    # Performance counters
    attempts = 0
    heralds = 0
    useful_pairs = 0
    fidelity_failures = 0

    measurements = 0
    calibrations = 0
    constraint_violations = 0

    # Recovery diagnostics
    recovery_stalls = 0
    idle_time = 0.0
    consecutive_maintenance = 0
    max_streak_observed = 0
    max_maintenance = 8

    # Controller state:
    # after calibration, take a measurement
    # before reconsidering the next action.
    measure_after_calibration = False

    stopped_early = False

    while t < T_end:

        remaining = T_end - t

        reliability = predicted_reliability(
            m, v, Q, attempt_dt, sigma, F_min
        )

        # ----------------------------------
        # Action selection
        # ----------------------------------

        if consecutive_maintenance >= max_maintenance:
            # Recovery stalled.
            # Do not attempt an inadmissible pair.
            action = "idle"
            dt = min(idle_dt, remaining)

        elif measure_after_calibration:
            action = "measure"
            dt = measure_dt

        elif reliability >= reliability_target:
            action = "attempt"
            dt = attempt_dt

        else:
            # Initial recovery heuristic:
            # significant estimated misalignment
            # motivates calibration.
            if (
                abs(m) >= 3.0
                and abs(m) >= 1.5 * np.sqrt(v)
            ):
                action = "calibrate"
                dt = calibrate_dt
            else:
                action = "measure"
                dt = measure_dt

        # ----------------------------------
        # Handle end-of-horizon conditions
        # ----------------------------------

        if dt > remaining + 1e-9:
            # No selected action fits.
            # Advance physical drift through the
            # remaining time without generating pairs.
            delta, m, v = advance_state(
                delta, m, v, Q, remaining, rng
            )
            idle_time += remaining
            t = T_end
            break

        # ----------------------------------
        # Execute action
        # ----------------------------------

        if action == "idle":

            delta, m, v = advance_state(
                delta, m, v, Q, dt, rng
            )

            idle_time += dt
            recovery_stalls += 1

            # Reset maintenance streak to allow
            # another recovery attempt.
            consecutive_maintenance = 0

            # The link is not automatically
            # considered reliable after idling.

        elif action == "measure":

            delta, m, v = advance_state(
                delta, m, v, Q, dt, rng
            )

            y, m, v = measure_and_update(
                delta, m, v, r, rng
            )

            measurements += 1
            consecutive_maintenance += 1

            measure_after_calibration = False

        elif action == "calibrate":

            delta, m, v = calibrate(
                delta, m, v, Q, dt, a, rng
            )

            calibrations += 1
            consecutive_maintenance += 1

            # Explicit recovery sequence:
            # calibration must be followed
            # by measurement.
            measure_after_calibration = True

        elif action == "attempt":

            # Verify chance constraint before
            # executing the attempt.
            if reliability < reliability_target:
                constraint_violations += 1
                raise RuntimeError(
                    "Attempt violates reliability target"
                )

            (
                delta,
                m,
                v,
                heralded,
                F,
                useful
            ) = execute_attempt(
                delta, m, v, Q, dt,
                sigma, p_h, F_min, rng
            )

            attempts += 1
            heralds += int(heralded)
            useful_pairs += int(useful)

            if heralded and not useful:
                fidelity_failures += 1

            consecutive_maintenance = 0

        max_streak_observed = max(
            max_streak_observed,
            consecutive_maintenance
        )

        t += dt

    throughput = useful_pairs / (T_end * 1e-6)

    return {
        "attempts": attempts,
        "heralds": heralds,
        "useful_pairs": useful_pairs,
        "fidelity_failures": fidelity_failures,
        "measurements": measurements,
        "calibrations": calibrations,
        "constraint_violations": constraint_violations,
        "stopped_early": stopped_early,
        "elapsed_us": t,
        "throughput_pairs_per_sec": throughput,
        "max_maintenance_streak": max_streak_observed,
        "recovery_stalls": recovery_stalls,
        "idle_us": idle_time,
        "final_m": m,
        "final_v": v,
        "final_reliability": predicted_reliability(
            m, v, Q, attempt_dt, sigma, F_min
        )
    }


#def run_reactive_fixed(seed=42, n_attempts=8,reliability_target=0.99):
def run_reactive_fixed(seed=42, n_attempts=100, reliability_target=0.99, Q=0.09, r=2.0):
    rng = np.random.default_rng(seed)

    # Same physical model
    T_end = 1000.0
    #Q = 0.09
    #r = 2.0
    a = 2.0
    sigma = 10.0
    p_h = 0.25
    F_min = 0.95

    durations = {
        "measure": 5.0,
        "calibrate": 15.0,
        "attempt": 10.0
    }

    # Same initial conditions
    t = 0.0
    delta = 5.0
    m = 4.0
    v = 4.0

    attempts = 0
    heralds = 0
    useful_pairs = 0
    fidelity_failures = 0
    measurements = 0
    calibrations = 0
    skipped_slots = 0
    rejected_slots = 0
    recovery_actions = 0
    constraint_violations = 0

    # Normal operation follows a periodic schedule.
    schedule = (
        ["measure", "calibrate", "measure"]
        + ["attempt"] * n_attempts
    )

    action_index = 0
    in_recovery = False
    measure_after_calibration = False
    recovery_streak = 0
    max_recovery = 8

    while t < T_end:

        scheduled_action = schedule[action_index]

        # --------------------------------------
        # Select action
        # --------------------------------------

        if in_recovery:

            if recovery_streak >= max_recovery:
                # Give up on this attempt slot.
                # Advance time and drift normally.
                dt = durations["attempt"]

                if t + dt > T_end:
                    break

                delta, m, v = advance_state(
                    delta, m, v, Q, dt, rng
                )

                t += dt
                skipped_slots += 1

                in_recovery = False
                measure_after_calibration = False
                recovery_streak = 0

                action_index = (
                    action_index + 1
                ) % len(schedule)

                continue

            if measure_after_calibration:
                action = "measure"

            else:
                reliability = predicted_reliability(
                    m, v, Q,
                    durations["attempt"],
                    sigma, F_min
                )

                if reliability >= reliability_target:
                    # Recovery succeeded.
                    in_recovery = False
                    recovery_streak = 0
                    continue

                if (
                    abs(m) >= 3.0
                    and abs(m) >= 1.5 * np.sqrt(v)
                ):
                    action = "calibrate"
                else:
                    action = "measure"

        else:
            action = scheduled_action

            # Check admission before a scheduled
            # entanglement attempt.
            if action == "attempt":

                reliability = predicted_reliability(
                    m, v, Q,
                    durations["attempt"],
                    sigma, F_min
                )

                if reliability < reliability_target:
                    rejected_slots += 1
                    in_recovery = True
                    recovery_streak = 0
                    continue

        dt = durations[action]

        if t + dt > T_end:
            break

        # --------------------------------------
        # Execute action
        # --------------------------------------

        if action == "measure":

            delta, m, v = advance_state(
                delta, m, v, Q, dt, rng
            )

            _, m, v = measure_and_update(
                delta, m, v, r, rng
            )

            measurements += 1

            if in_recovery:
                recovery_actions += 1
                recovery_streak += 1
                measure_after_calibration = False

        elif action == "calibrate":

            delta, m, v = calibrate(
                delta, m, v, Q, dt, a, rng
            )

            calibrations += 1

            if in_recovery:
                recovery_actions += 1
                recovery_streak += 1
                measure_after_calibration = True

        elif action == "attempt":

            reliability = predicted_reliability(
                m, v, Q, dt, sigma, F_min
            )

            if reliability < reliability_target:
                constraint_violations += 1
                raise RuntimeError(
                    "Inadmissible entanglement attempt"
                )

            delta, m, v, heralded, F, useful = (
                execute_attempt(
                    delta, m, v, Q, dt,
                    sigma, p_h, F_min, rng
                )
            )

            attempts += 1
            heralds += int(heralded)
            useful_pairs += int(useful)

            if heralded and not useful:
                fidelity_failures += 1

        t += dt

        # Advance normal schedule only when
        # its scheduled action was completed.
        # Recovery actions leave the slot pending.
        if not in_recovery:
            action_index = (
                action_index + 1
            ) % len(schedule)

    throughput = useful_pairs / (T_end * 1e-6)

    assert t <= T_end + 1e-9
    assert heralds == useful_pairs + fidelity_failures
    assert constraint_violations == 0

    return {
        "attempts": attempts,
        "heralds": heralds,
        "useful_pairs": useful_pairs,
        "fidelity_failures": fidelity_failures,
        "measurements": measurements,
        "calibrations": calibrations,
        "rejected_slots": rejected_slots,
        "skipped_slots": skipped_slots,
        "recovery_actions": recovery_actions,
        "constraint_violations": constraint_violations,
        "elapsed_us": t,
        "throughput_pairs_per_sec": throughput
    }


def evaluate_baseline(n_runs=1000):
    results = [
        run_fixed_baseline(seed=i)
        for i in range(n_runs)
    ]

    metrics = [
        "attempts",
        "heralds",
        "useful_pairs",
        "throughput_pairs_per_sec"
    ]

    print(f"\nResults across {n_runs} independent runs")
    print("-" * 50)

    for metric in metrics:
        values = np.array([r[metric] for r in results])

        print(
            f"{metric:30s}"
            f"mean={values.mean():10.2f}  "
            f"std={values.std(ddof=1):10.2f}"
        )

    expected_heralds = 80 * 0.25
    print(f"\nExpected heralds per run: {expected_heralds:.2f}")



def compare_controllers(n_runs=1000):

    fixed_results = [
        run_fixed_baseline(seed=i)
        for i in range(n_runs)
    ]

    adaptive_results = [
        run_adaptive(seed=i, reliability_target=0.99)
        for i in range(n_runs)
    ]

    metrics = [
        "attempts",
        "heralds",
        "useful_pairs",
        "fidelity_failures",
        "throughput_pairs_per_sec"
    ]

    print("\nFIXED vs ADAPTIVE")
    print("-" * 65)
    print(f"{'Metric':30s} {'Fixed':>14s} {'Adaptive':>14s}")

    for metric in metrics:

        fixed = np.array([
            result[metric] for result in fixed_results
        ])

        adaptive = np.array([
            result[metric] for result in adaptive_results
        ])

        print(
            f"{metric:30s}"
            f"{fixed.mean():14.2f}"
            f"{adaptive.mean():14.2f}"
        )

    print("\nADAPTIVE MAINTENANCE")

    for metric in ["measurements", "calibrations",
                   "constraint_violations"]:

        values = np.array([
            result[metric] for result in adaptive_results
        ])

        print(f"{metric:30s}{values.mean():14.2f}")

    stopped = sum(
        result["stopped_early"]
        for result in adaptive_results
    )

    print(f"\nEarly stops: {stopped}/{n_runs}")

    early_runs = [
    r for r in adaptive_results
    if r["stopped_early"]
    ]

    print("\nEARLY STOP DIAGNOSTICS")

    if early_runs:
        for metric in [
            "elapsed_us",
            "max_maintenance_streak",
            "final_m",
            "final_v",
            "final_reliability"
        ]:
            values = np.array([r[metric] for r in early_runs])
            print(f"{metric:30s} mean={values.mean():.3f}")

    print("\nSAMPLE EARLY-STOP STATES")

    for result in early_runs[:10]:
        print(
            f"t={result['elapsed_us']:6.1f} us  "
            f"m={result['final_m']:7.3f} ps  "
            f"v={result['final_v']:7.3f} ps^2  "
            f"P={result['final_reliability']:.4f}"
        )


    print("\nSTRUCTURED RECOVERY DIAGNOSTICS")

    for metric in ["recovery_stalls", "idle_us"]:
        values = np.array([
            result[metric] for result in adaptive_results
        ])

        print(
            f"{metric:30s}"
            f"mean={values.mean():10.2f}  "
            f"max={values.max():10.2f}"
        )

    print(
        "Runs with recovery stalls:",
        sum(r["recovery_stalls"] > 0 for r in adaptive_results),
        "/",
        n_runs
    )

    violations = np.array([
        r["predicted_violations"]
        for r in fixed_results
    ])

    reliabilities = np.array([
        r["mean_predicted_reliability"]
        for r in fixed_results
    ])

    print("\nFIXED BASELINE RELIABILITY AUDIT")

    print(f"Mean predicted violations/run: {violations.mean():.2f}")

    print(
        f"Fraction of attempts below 99%: "
        f"{violations.sum() / sum(r["attempts"] for r in fixed_results)}"


    )

    print(
        f"Mean predicted reliability: "
        f"{reliabilities.mean():.2%}"
    )

def sweep_fixed_schedules(n_runs=1000):
    print("\nFIXED SCHEDULE SWEEP")
    print("-" * 75)

    for n in [2, 4, 6, 8, 10]:
        results = [
            run_fixed_baseline(seed=i, n_attempts=n)
            for i in range(n_runs)
        ]

        throughput = np.mean([
            r["throughput_pairs_per_sec"] for r in results
        ])

        total_violations = sum(
            r["predicted_violations"] for r in results
        )

        total_attempts = sum(
            r["attempts"] for r in results
        )

        violation_rate = total_violations / total_attempts

        print(
            f"Attempts/cycle={n:2d}  "
            f"Throughput={throughput:9.2f}/s  "
            f"Violation rate={violation_rate:7.2%}"
        )


def sweep_gated_fixed(n_runs=1000):
    print("\nRELIABILITY-GATED FIXED SCHEDULE SWEEP")
    print("-" * 90)

    for n in [2, 4, 6, 8, 10]:
        results = [
            run_gated_fixed(
                seed=i,
                n_attempts=n,
                reliability_target=0.99
            )
            for i in range(n_runs)
        ]

        throughput = np.mean([
            x["throughput_pairs_per_sec"]
            for x in results
        ])

        attempts = sum(x["attempts"] for x in results)
        skipped = sum(x["skipped_attempts"] for x in results)
        slots = attempts + skipped

        skip_fraction = skipped / slots if slots else 0

        failures = sum(
            x["fidelity_failures"] for x in results
        )

        heralds = sum(x["heralds"] for x in results)

        failure_fraction = (
            failures / heralds if heralds else 0
        )

        print(
            f"n={n:2d}  "
            f"Throughput={throughput:9.2f}/s  "
            f"Attempts/run={attempts/n_runs:6.2f}  "
            f"Skipped={skip_fraction:7.2%}  "
            f"Fidelity failures={failure_fraction:7.2%}"
        )


def sweep_reactive_fixed(n_runs=1000):

    print("\nREACTIVE PERIODIC BASELINE SWEEP")
    print("-" * 90)

    #for n in [2, 4, 6, 8, 10]:
    for n in [10, 15, 20, 30, 50, 100]:

        results = [
            run_reactive_fixed(
                seed=i,
                n_attempts=n,
                reliability_target=0.99
            )
            for i in range(n_runs)
        ]

        def avg(key):
            return np.mean([
                result[key] for result in results
            ])

        print(
            f"n={n:2d}  "
            f"Throughput={avg('throughput_pairs_per_sec'):9.2f}/s  "
            f"Attempts={avg('attempts'):6.2f}  "
            f"Measurements={avg('measurements'):6.2f}  "
            f"Calibrations={avg('calibrations'):6.2f}  "
            f"Recovery={avg('recovery_actions'):6.2f}  "
            f"Skipped={avg('skipped_slots'):6.2f}"
        )

    print("\nADAPTIVE V3 REFERENCE")

    adaptive = [
        run_adaptive(seed=i, reliability_target=0.99)
        for i in range(n_runs)
    ]

    print(
        "Throughput:",
        np.mean([
            result["throughput_pairs_per_sec"]
            for result in adaptive
        ])
    )


def compare_statistically(n_runs=1000):
    reactive = np.array([
        run_reactive_fixed(seed=i, n_attempts=100)[
            "throughput_pairs_per_sec"
        ]
        for i in range(n_runs)
    ])

    adaptive = np.array([
        run_adaptive(seed=i)[
            "throughput_pairs_per_sec"
        ]
        for i in range(n_runs)
    ])

    # Compare sample means
    mean_reactive = reactive.mean()
    mean_adaptive = adaptive.mean()

    # Difference across matching seed indices
    differences = adaptive - reactive
    mean_diff = differences.mean()

    # Sample variability and standard error
    std_diff = differences.std(ddof=1)
    se_diff = std_diff / np.sqrt(n_runs)

    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff

    print("\nSTATISTICAL COMPARISON")
    print("-" * 55)
    print(f"Reactive mean:     {mean_reactive:.2f}")
    print(f"Adaptive mean:     {mean_adaptive:.2f}")
    print(f"Mean difference:   {mean_diff:.2f}")
    print(f"Std difference:    {std_diff:.2f}")
    print(f"95% CI difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(
        f"Relative difference: "
        f"{100 * mean_diff / mean_reactive:.2f}%"
    )


def sweep_drift_rates(n_runs=1000):
    Q_values = [0.25, 0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]

    print("\nDRIFT-RATE SENSITIVITY")
    print("-" * 88)
    print(
        f"{'Q':>7} {'Reactive':>12} {'Adaptive':>12} "
        f"{'Difference':>12} {'95% CI':>30}"
    )

    for Q in Q_values:
        reactive = np.array([
            run_reactive_fixed(
                seed=i, n_attempts=100, Q=Q
            )["throughput_pairs_per_sec"]
            for i in range(n_runs)
        ])

        adaptive = np.array([
            run_adaptive(
                seed=i, Q=Q
            )["throughput_pairs_per_sec"]
            for i in range(n_runs)
        ])

        differences = adaptive - reactive
        mean_diff = differences.mean()
        se = differences.std(ddof=1) / np.sqrt(n_runs)

        lower = mean_diff - 1.96 * se
        upper = mean_diff + 1.96 * se

        print(
            f"{Q:7.2f} "
            f"{reactive.mean():12.2f} "
            f"{adaptive.mean():12.2f} "
            f"{mean_diff:12.2f} "
            f"[{lower:9.2f}, {upper:9.2f}]"
        )


def sweep_sensor_precision(n_runs=1000):
    r_values = [1.0, 2.0, 3.0, 4.0]
    Q_values = np.arange(0.05, 0.81, 0.05)

    print("\nSENSOR PRECISION AND DRIFT SWEEP")
    print("-" * 75)
    print(
        f"{'r (ps)':>8} {'Q':>8} "
        f"{'Reactive/s':>14} {'Adaptive/s':>14}"
    )

    for r in r_values:
        for Q in Q_values:
            reactive = [
                run_reactive_fixed(
                    seed=i, n_attempts=100, Q=Q, r=r
                )["throughput_pairs_per_sec"]
                for i in range(n_runs)
            ]

            adaptive = [
                run_adaptive(
                    seed=i, Q=Q, r=r
                )["throughput_pairs_per_sec"]
                for i in range(n_runs)
            ]

            print(
                f"{r:8.2f} {Q:8.2f} "
                f"{np.mean(reactive):14.2f} "
                f"{np.mean(adaptive):14.2f}"
            )


if __name__ == "__main__":
    sweep_sensor_precision(n_runs=1000)

