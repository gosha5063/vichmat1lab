from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Func = Callable[[float], float]


@dataclass(frozen=True)
class IntegralResult:
    value: float
    n: int
    h: float
    method: str


def midpoint(f: Func, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    xs = a + (np.arange(n) + 0.5) * h
    return h * float(np.sum([f(float(x)) for x in xs]))


def trapezoid(f: Func, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    xs = a + np.arange(n + 1) * h
    ys = np.array([f(float(x)) for x in xs], dtype=float)
    return h * (0.5 * ys[0] + np.sum(ys[1:-1]) + 0.5 * ys[-1])


def simpson(f: Func, a: float, b: float, n: int) -> float:
    if n % 2 != 0:
        raise ValueError("Simpson requires even n")
    h = (b - a) / n
    xs = a + np.arange(n + 1) * h
    ys = np.array([f(float(x)) for x in xs], dtype=float)
    return h / 3.0 * (ys[0] + ys[-1] + 4.0 * np.sum(ys[1:-1:2]) + 2.0 * np.sum(ys[2:-1:2]))


def step_from_second_derivative(
    *, m2: float, a: float, b: float, eps: float, order: int
) -> tuple[int, float]:
    """
    Оценка шага по максимуму второй производной.
    midpoint:  |R| <= (b-a) h^2 / 24 * M2
    trapezoid: |R| <= (b-a) h^2 / 12 * M2
    """
    length = b - a
    if order == 2:
        h = np.sqrt(24.0 * eps / (length * m2))
    elif order == 1:
        h = np.sqrt(12.0 * eps / (length * m2))
    else:
        raise ValueError("order must be 1 (trapezoid) or 2 (midpoint)")
    n = max(1, int(np.ceil(length / h)))
    h = length / n
    return n, float(h)


def integrate_with_runge(
    method: str,
    f: Func,
    a: float,
    b: float,
    *,
    eps: float,
    start_n: int = 2,
    max_refine: int = 20,
) -> IntegralResult:
    if method == "trapezoid":
        compute = trapezoid
        denom = 3.0
    elif method == "simpson":
        compute = simpson
        denom = 15.0
    else:
        raise ValueError("method must be 'trapezoid' or 'simpson'")

    n = max(2, start_n)
    if method == "simpson" and n % 2 != 0:
        n += 1

    i_coarse = compute(f, a, b, n)
    for _ in range(max_refine):
        i_fine = compute(f, a, b, 2 * n)
        err_est = abs(i_coarse - i_fine) / denom
        if err_est <= eps:
            return IntegralResult(
                value=i_fine,
                n=2 * n,
                h=(b - a) / (2 * n),
                method=method,
            )
        i_coarse = i_fine
        n *= 2

    return IntegralResult(
        value=i_coarse,
        n=n,
        h=(b - a) / n,
        method=method,
    )
