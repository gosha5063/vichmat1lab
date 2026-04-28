import math

import matplotlib.pyplot as plt
import numpy as np

from nonlinear_methods import bisection, localize_roots_by_sign_changes, sample_function


def f(x: float) -> float:
    # x^2 - 20 sin(x) = 0
    return x * x - 20.0 * math.sin(x)


def main():
    eps = 0.001

    # 1) График и локализация (делаем численно: ищем смены знака на разумном диапазоне)
    a_scan, b_scan = -10.0, 30.0
    intervals = localize_roots_by_sign_changes(f, a_scan, b_scan, n=8000)

    print("Локализация корней по смене знака на [{:.1f}, {:.1f}]:".format(a_scan, b_scan))
    for i, (a, b) in enumerate(intervals, 1):
        if a == b:
            print(f"  {i}) x = {a:.6f} (точка, f(x)=0)")
        else:
            print(f"  {i}) [{a:.6f}, {b:.6f}]")

    # для отчёта берём нетривиальный корень (не x=0)
    chosen = None
    for a, b in intervals:
        if a == b:
            continue
        mid = 0.5 * (a + b)
        if abs(mid) < 1e-6:
            continue
        # избегаем интервала вокруг 0
        if b <= 0.2:
            continue
        chosen = (a, b)
        break

    if chosen is None:
        raise RuntimeError("Не удалось выбрать интервал для нетривиального корня.")

    a0, b0 = chosen
    res = bisection(f, a0, b0, eps=eps)

    print("\nБисекция на интервале [{:.6f}, {:.6f}] при eps = {}:".format(a0, b0, eps))
    print("  x* = {:.10f}".format(res.x))
    print("  iterations =", res.iters)
    print("  last_step  = {:.3e}".format(res.last_step))
    print("  |f(x*)|    = {:.3e}".format(abs(f(res.x))))

    # 2) Рисуем график
    xs, ys = sample_function(f, a_scan, b_scan, n=4000)
    plt.figure(figsize=(9, 4))
    plt.axhline(0, color="black", linewidth=1)
    plt.plot(xs, ys, label=r"$f(x)=x^2-20\sin x$")
    plt.ylim(-50, 200)
    plt.grid(True, alpha=0.3)
    plt.scatter([res.x], [f(res.x)], color="red", zorder=3, label=f"root ~ {res.x:.4f}")
    plt.title("Локализация корней и найденный корень (бисекция)")
    plt.legend()
    out = "p1_bisection_plot.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print("\nГрафик сохранён в файл:", out)


if __name__ == "__main__":
    main()

