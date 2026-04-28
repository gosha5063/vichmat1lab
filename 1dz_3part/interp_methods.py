from __future__ import annotations

import numpy as np


def barycentric_weights(x: np.ndarray) -> np.ndarray:
    """
    Веса для барицентрической формулы Лагранжа:
        w_i = 1 / Π_{j!=i} (x_i - x_j)
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    w = np.ones(n, dtype=float)
    for i in range(n):
        diffs = x[i] - np.delete(x, i)
        w[i] = 1.0 / np.prod(diffs)
    return w


def lagrange_barycentric_eval(x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Значение интерполяционного полинома Лагранжа в точках x_eval
    по барицентрической формуле (устойчивее, чем явное суммирование базисов).
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    w = barycentric_weights(x_nodes)

    out = np.empty_like(x_eval, dtype=float)
    for k, x in enumerate(x_eval):
        # если попали в узел — возвращаем точное значение
        idx = np.where(np.isclose(x_nodes, x, atol=0.0, rtol=0.0))[0]
        if idx.size:
            out[k] = y_nodes[int(idx[0])]
            continue

        diffs = x - x_nodes
        num = np.sum((w * y_nodes) / diffs)
        den = np.sum(w / diffs)
        out[k] = num / den
    return out


def cubic_spline_natural(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Натуральный кубический сплайн: S''(x1)=S''(xn)=0.
    Возвращает коэффициенты для каждого интервала [x_i, x_{i+1}]:
        S_i(t) = a_i + b_i*(t-x_i) + c_i*(t-x_i)^2 + d_i*(t-x_i)^3
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 nodes for cubic spline")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")

    h = np.diff(x)

    # Решаем систему для c (вторая производная /2, в классической записи)
    A = np.zeros((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)

    # натуральные граничные условия
    A[0, 0] = 1.0
    A[n - 1, n - 1] = 1.0

    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    c = np.linalg.solve(A, rhs)

    a = y[:-1].copy()
    b = np.empty(n - 1, dtype=float)
    d = np.empty(n - 1, dtype=float)
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    return a, b, c[:-1], d


def cubic_spline_eval(
    x_nodes: np.ndarray, coeffs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], x_eval: np.ndarray
) -> np.ndarray:
    x_nodes = np.asarray(x_nodes, dtype=float)
    a, b, c, d = coeffs
    x_eval = np.asarray(x_eval, dtype=float)

    n = x_nodes.size
    out = np.empty_like(x_eval, dtype=float)

    for k, x in enumerate(x_eval):
        if x <= x_nodes[0]:
            i = 0
        elif x >= x_nodes[-1]:
            i = n - 2
        else:
            i = int(np.searchsorted(x_nodes, x) - 1)
            i = max(0, min(i, n - 2))
        dx = x - x_nodes[i]
        out[k] = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    return out

