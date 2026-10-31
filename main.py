
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

def run_fixed_baseline(seed=42):
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

    # Initial state
    t = 0.0
    delta = 5.0
    m = 4.0
    v = 4.0

    attempts = 0
    heralds = 0
    useful_pairs = 0

    schedule = ["measure", "calibrate"] + ["attempt"] * 8

    while t < T_end:
        for action in schedule:

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

            elif action == "calibrate":
                delta, m, v = calibrate(
                    delta, m, v, Q, dt, a, rng
                )


            elif action == "attempt":
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

        if t >= T_end:
            break

    throughput = useful_pairs / (T_end * 1e-6)

    return {
        "attempts": attempts,
        "heralds": heralds,
        "useful_pairs": useful_pairs,
        "throughput_pairs_per_sec": throughput,
        "fidelity_failures": fidelity_failures
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




def run_adaptive(seed=42, reliability_target=0.99):
    rng = np.random.default_rng(seed)

    # Physical parameters: unchanged
    T_end = 1000.0
    Q = 0.09
    r = 2.0
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

if __name__ == "__main__":
    compare_controllers(n_runs=1000)


