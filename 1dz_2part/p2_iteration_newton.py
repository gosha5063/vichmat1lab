import math

import matplotlib.pyplot as plt
import numpy as np

from nonlinear_methods import fixed_point_iteration, localize_roots_by_sign_changes, newton, sample_function


def f(x: float) -> float:
    # x^3 - 3x - 3 = 0
    return x**3 - 3.0 * x - 3.0


def df(x: float) -> float:
    return 3.0 * x**2 - 3.0


def phi(x: float) -> float:
    # Выбор для простой итерации: x = (3x + 3)^{1/3}
    return (3.0 * x + 3.0) ** (1.0 / 3.0)


def dphi(x: float) -> float:
    return (1.0) / ((3.0 * x + 3.0) ** (2.0 / 3.0))


def main():
    eps = 0.001

    # Локализация по графику/смене знака
    a_scan, b_scan = -5.0, 5.0
    intervals = localize_roots_by_sign_changes(f, a_scan, b_scan, n=4000)
    print("Локализация корня по смене знака на [{:.1f}, {:.1f}]:".format(a_scan, b_scan))
    for i, (a, b) in enumerate(intervals, 1):
        if a == b:
            print(f"  {i}) x = {a:.6f} (точка, f(x)=0)")
        else:
            print(f"  {i}) [{a:.6f}, {b:.6f}]")

    # У уравнения x^3-3x-3=0 один действительный корень, возьмем первый интервал смены знака
    seg = None
    for a, b in intervals:
        if a != b:
            seg = (a, b)
            break
    if seg is None:
        raise RuntimeError("Не найден интервал локализации.")

    a0, b0 = seg
    x0 = 0.5 * (a0 + b0)

    # Проверка условия сходимости простой итерации на сегменте: max |phi'(x)| < 1
    xs = np.linspace(a0, b0, 2001)
    q = float(np.max(np.abs([dphi(float(x)) for x in xs])))

    print("\nВыбран интервал [{:.6f}, {:.6f}], старт x0 = {:.6f}".format(a0, b0, x0))
    print("Оценка q = max|phi'(x)| на интервале:", "{:.6f}".format(q))

    res_it = fixed_point_iteration(phi, x0, eps=eps, max_iter=1_000_000)
    res_n = newton(f, df, x0, eps=eps, max_iter=1_000_000)

    print("\nМетод простой итерации (eps = {}):".format(eps))
    print("  x* =", "{:.10f}".format(res_it.x))
    print("  iterations =", res_it.iters)
    print("  last_step  =", "{:.3e}".format(res_it.last_step))
    print("  |f(x*)|    =", "{:.3e}".format(abs(f(res_it.x))))

    print("\nМетод Ньютона (eps = {}):".format(eps))
    print("  x* =", "{:.10f}".format(res_n.x))
    print("  iterations =", res_n.iters)
    print("  last_step  =", "{:.3e}".format(res_n.last_step))
    print("  |f(x*)|    =", "{:.3e}".format(abs(f(res_n.x))))

    # График
    xs_plot, ys_plot = sample_function(f, a_scan, b_scan, n=2000)
    plt.figure(figsize=(9, 4))
    plt.axhline(0, color="black", linewidth=1)
    plt.plot(xs_plot, ys_plot, label=r"$f(x)=x^3-3x-3$")
    plt.grid(True, alpha=0.3)
    plt.scatter([res_it.x], [f(res_it.x)], color="orange", zorder=3, label=f"iter ~ {res_it.x:.4f}")
    plt.scatter([res_n.x], [f(res_n.x)], color="red", zorder=3, label=f"newton ~ {res_n.x:.4f}")
    plt.title("Локализация и найденный корень (итерации / Ньютон)")
    plt.legend()
    out = "p2_iteration_newton_plot.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print("\nГрафик сохранён в файл:", out)


if __name__ == "__main__":
    main()

