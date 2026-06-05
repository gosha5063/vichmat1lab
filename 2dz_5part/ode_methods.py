from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Rhs = Callable[[float, float], float]


@dataclass(frozen=True)
class OdeResult:
    xs: np.ndarray
    ys: np.ndarray
    n_steps: int
    method: str


def euler_cauchy_step(f: Rhs, x: float, y: float, h: float) -> float:
    k1 = f(x, y)
    y_pred = y + h * k1
    k2 = f(x + h, y_pred)
    return y + 0.5 * h * (k1 + k2)


def rk4_step(f: Rhs, x: float, y: float, h: float) -> float:
    k1 = f(x, y)
    k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(x + h, y + h * k3)
    return y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _advance_one_step(stepper, f: Rhs, x: float, y: float, h: float) -> float:
    return stepper(f, x, y, h)


def solve_adaptive(
    stepper,
    f: Rhs,
    x0: float,
    xn: float,
    y0: float,
    *,
    eps: float,
    h0: float,
    runge_divisor: float,
    method_name: str,
    h_min: float = 1e-6,
    h_max: float = 0.5,
) -> OdeResult:
    xs = [x0]
    ys = [y0]
    x = x0
    y = y0
    h = h0
    n_steps = 0

    while x < xn - 1e-15:
        h = min(h, xn - x)
        y_full = _advance_one_step(stepper, f, x, y, h)
        y_half = _advance_one_step(
            stepper, f, x, y, 0.5 * h
        )
        y_half = _advance_one_step(
            stepper, f, x + 0.5 * h, y_half, 0.5 * h
        )
        err = abs(y_full - y_half) / runge_divisor

        if err <= eps:
            x += h
            y = y_half
            xs.append(x)
            ys.append(y)
            n_steps += 1
            if err < eps / 10.0:
                h = min(h_max, 2.0 * h)
        else:
            h = max(h_min, 0.5 * h)
            if h <= h_min:
                x += h
                y = y_half
                xs.append(x)
                ys.append(y)
                n_steps += 1

    return OdeResult(
        xs=np.array(xs, dtype=float),
        ys=np.array(ys, dtype=float),
        n_steps=n_steps,
        method=method_name,
    )
