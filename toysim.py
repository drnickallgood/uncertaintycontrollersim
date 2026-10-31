
import numpy as np

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

            t += dt

        if t >= T_end:
            break

    throughput = useful_pairs / (T_end * 1e-6)

    return {
        "attempts": attempts,
        "heralds": heralds,
        "useful_pairs": useful_pairs,
        "throughput_pairs_per_sec": throughput
    }


if __name__ == "__main__":
    print(run_fixed_baseline())
