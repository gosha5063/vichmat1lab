from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


Func = Callable[[float], float]


@dataclass(frozen=True)
class RootResult:
    x: float
    iters: int
    converged: bool
    last_step: float
    last_residual: float


def localize_roots_by_sign_changes(
    f: Func, a: float, b: float, *, n: int = 2000
) -> list[tuple[float, float]]:
    """
    Грубая локализация корней на [a,b] по смене знака f(x).
    Возвращает интервалы [x_i, x_{i+1}] где f меняет знак или попадает в 0.
    """
    xs = np.linspace(a, b, n + 1, dtype=float)
    fs = np.array([f(float(x)) for x in xs], dtype=float)

    intervals: list[tuple[float, float]] = []
    for i in range(n):
        f1, f2 = fs[i], fs[i + 1]
        if np.isnan(f1) or np.isnan(f2) or np.isinf(f1) or np.isinf(f2):
            continue
        if np.isclose(f1, 0.0):
            intervals.append((float(xs[i]), float(xs[i])))
            continue
        if f1 * f2 < 0:
            intervals.append((float(xs[i]), float(xs[i + 1])))
        elif np.isclose(f2, 0.0):
            intervals.append((float(xs[i + 1]), float(xs[i + 1])))
    # убираем возможные дубли/слипания
    merged: list[tuple[float, float]] = []
    for l, r in intervals:
        if not merged:
            merged.append((l, r))
            continue
        pl, pr = merged[-1]
        if np.isclose(l, pr) and (l == r or pl == pr):
            merged[-1] = (pl, r)
        else:
            merged.append((l, r))
    return merged


def bisection(
    f: Func, a: float, b: float, *, eps: float, max_iter: int = 10_000
) -> RootResult:
    """
    Метод половинного деления (бисекции).
    Критерий остановки: (b-a)/2 <= eps.
    Требует f(a)*f(b) <= 0.
    """
    fa, fb = f(a), f(b)
    if np.isnan(fa) or np.isnan(fb):
        raise ValueError("f(a) or f(b) is NaN")
    if fa == 0.0:
        return RootResult(x=float(a), iters=0, converged=True, last_step=0.0, last_residual=0.0)
    if fb == 0.0:
        return RootResult(x=float(b), iters=0, converged=True, last_step=0.0, last_residual=0.0)
    if fa * fb > 0:
        raise ValueError("Bisection requires f(a) and f(b) to have opposite signs (or zero).")

    left, right = float(a), float(b)
    fleft, fright = float(fa), float(fb)
    it = 0
    while it < max_iter:
        mid = 0.5 * (left + right)
        fmid = float(f(mid))
        half_len = 0.5 * (right - left)
        if half_len <= eps:
            return RootResult(
                x=float(mid),
                iters=it + 1,
                converged=True,
                last_step=float(half_len),
                last_residual=float(abs(fmid)),
            )
        if np.isclose(fmid, 0.0):
            return RootResult(
                x=float(mid),
                iters=it + 1,
                converged=True,
                last_step=float(half_len),
                last_residual=0.0,
            )
        if fleft * fmid < 0:
            right, fright = mid, fmid
        else:
            left, fleft = mid, fmid
        it += 1

    mid = 0.5 * (left + right)
    return RootResult(
        x=float(mid),
        iters=it,
        converged=False,
        last_step=float(0.5 * (right - left)),
        last_residual=float(abs(f(mid))),
    )


def fixed_point_iteration(
    phi: Func,
    x0: float,
    *,
    eps: float,
    max_iter: int = 100_000,
) -> RootResult:
    """
    Метод простой итерации x_{k+1} = phi(x_k)
    Критерий остановки: |x_{k+1} - x_k| <= eps.
    """
    x_prev = float(x0)
    for k in range(1, max_iter + 1):
        x_next = float(phi(x_prev))
        step = abs(x_next - x_prev)
        if step <= eps:
            return RootResult(
                x=x_next,
                iters=k,
                converged=True,
                last_step=float(step),
                last_residual=float("nan"),
            )
        x_prev = x_next
    return RootResult(
        x=x_prev,
        iters=max_iter,
        converged=False,
        last_step=float("nan"),
        last_residual=float("nan"),
    )


def newton(
    f: Func,
    df: Func,
    x0: float,
    *,
    eps: float,
    max_iter: int = 100_000,
) -> RootResult:
    """
    Метод Ньютона: x_{k+1} = x_k - f(x_k)/f'(x_k)
    Критерий остановки: |x_{k+1}-x_k| <= eps.
    """
    x_prev = float(x0)
    for k in range(1, max_iter + 1):
        fval = float(f(x_prev))
        dval = float(df(x_prev))
        if np.isclose(dval, 0.0):
            return RootResult(
                x=x_prev,
                iters=k,
                converged=False,
                last_step=float("inf"),
                last_residual=float(abs(fval)),
            )
        x_next = x_prev - fval / dval
        step = abs(x_next - x_prev)
        if step <= eps:
            return RootResult(
                x=float(x_next),
                iters=k,
                converged=True,
                last_step=float(step),
                last_residual=float(abs(f(x_next))),
            )
        x_prev = float(x_next)
    return RootResult(
        x=x_prev,
        iters=max_iter,
        converged=False,
        last_step=float("nan"),
        last_residual=float(abs(f(x_prev))),
    )


def sample_function(f: Func, a: float, b: float, *, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(a, b, n + 1, dtype=float)
    ys = np.array([f(float(x)) for x in xs], dtype=float)
    return xs, ys

