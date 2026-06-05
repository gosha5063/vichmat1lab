from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from ode_methods import euler_cauchy_step, rk4_step, solve_adaptive

X0 = 0.0
XN = 1.5
Y0 = math.log(2.0)
EPS = 1e-3
H0 = 0.1


def f(x: float, y: float) -> float:
    return math.exp(x - y)


def y_exact(x: float) -> float:
    return math.log(1.0 + math.exp(x))


def main() -> None:
    ec = solve_adaptive(
        euler_cauchy_step,
        f,
        X0,
        XN,
        Y0,
        eps=EPS,
        h0=H0,
        runge_divisor=3.0,
        method_name="euler_cauchy",
    )
    rk = solve_adaptive(
        rk4_step,
        f,
        X0,
        XN,
        Y0,
        eps=EPS,
        h0=H0,
        runge_divisor=15.0,
        method_name="rk4",
    )

    y_end_exact = y_exact(XN)
    y_end_ec = ec.ys[-1]
    y_end_rk = rk.ys[-1]

    print("=== DZ5, variant 7: y' = e^(x-y), y(0)=ln2, [0;1.5] ===")
    print(f"Exact y({XN}) = {y_end_exact:.8f}")
    print()
    print(f"Euler-Cauchy: steps={ec.n_steps}, points={len(ec.xs)}, y_end={y_end_ec:.8f}, err={abs(y_end_ec-y_end_exact):.6e}")
    print(f"RK4:          steps={rk.n_steps}, points={len(rk.xs)}, y_end={y_end_rk:.8f}, err={abs(y_end_rk-y_end_exact):.6e}")

    x_plot = np.linspace(X0, XN, 400)
    y_plot = np.array([y_exact(float(x)) for x in x_plot])

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot, "k-", linewidth=2, label="Точное $y(x)=\\ln(1+e^x)$")
    plt.plot(ec.xs, ec.ys, "o-", markersize=4, label="Эйлер–Коши")
    plt.plot(rk.xs, rk.ys, "s-", markersize=4, label="Рунге–Кутта 4")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Задача Коши, вариант 7")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("p1_ode_plot.png", dpi=150)
    plt.close()
    print("\nГрафик сохранён: p1_ode_plot.png")


if __name__ == "__main__":
    main()
