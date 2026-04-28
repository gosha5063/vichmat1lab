import matplotlib.pyplot as plt
import numpy as np

from interp_methods import (
    cubic_spline_eval,
    cubic_spline_natural,
    lagrange_barycentric_eval,
)


def variant7_table():
    # Из variant.png (вариант 7)
    x = np.arange(1, 11, dtype=float)
    y = np.array([2.3, 3.71, 4.8, 5.9, 6.3, 6.25, 5.87, 4.82, 3.7, 2.2], dtype=float)
    return x, y


def main():
    x, y = variant7_table()

    # Лагранж (значения на сетке)
    x_grid = np.linspace(x[0], x[-1], 2000)
    y_l = lagrange_barycentric_eval(x, y, x_grid)

    # Кубический сплайн (натуральный)
    coeffs = cubic_spline_natural(x, y)
    y_s = cubic_spline_eval(x, coeffs, x_grid)

    # Сравнение в точках, не совпадающих с узлами (возьмем середины интервалов)
    x_mid = 0.5 * (x[:-1] + x[1:])
    y_l_mid = lagrange_barycentric_eval(x, y, x_mid)
    y_s_mid = cubic_spline_eval(x, coeffs, x_mid)

    print("Сравнение в серединах интервалов (x_mid):")
    print("| x | L(x) | S(x) | |S-L| |")
    print("|---:|---:|---:|---:|")
    for xi, li, si in zip(x_mid, y_l_mid, y_s_mid):
        print(f"| {xi:.2f} | {li:.6f} | {si:.6f} | {abs(si-li):.3e} |")

    # График
    plt.figure(figsize=(9, 4.5))
    plt.plot(x_grid, y_l, label="Полином Лагранжа", linewidth=1.6)
    plt.plot(x_grid, y_s, label="Кубический сплайн (натуральный)", linewidth=1.6)
    plt.scatter(x, y, color="black", zorder=3, label="Узлы (таблица)")
    plt.grid(True, alpha=0.3)
    plt.title("Вариант 7: интерполяция Лагранжем и кубическим сплайном")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    out = "p1_lagrange_spline_plot.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    print("\nГрафик сохранён в файл:", out)


if __name__ == "__main__":
    main()

