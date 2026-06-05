from __future__ import annotations

import math

from integration_methods import (
    integrate_with_runge,
    midpoint,
    step_from_second_derivative,
    trapezoid,
)

A = 0.0
B = 1.0
EPS = 1e-3


def f(x: float) -> float:
    return x * math.exp(x)


def f2(x: float) -> float:
    return math.exp(x) * (2.0 + x)


def exact_integral() -> float:
    # ∫ x e^x dx = e^x (x - 1)
    antideriv = lambda x: math.exp(x) * (x - 1.0)
    return antideriv(B) - antideriv(A)


def main() -> None:
    exact = exact_integral()
    m2 = max(f2(A), f2(B))

    n_mid, h_mid = step_from_second_derivative(
        m2=m2, a=A, b=B, eps=EPS, order=2
    )
    n_trap, h_trap = step_from_second_derivative(
        m2=m2, a=A, b=B, eps=EPS, order=1
    )

    i_mid = midpoint(f, A, B, n_mid)
    i_trap = trapezoid(f, A, B, n_trap)
    trap_runge = integrate_with_runge("trapezoid", f, A, B, eps=EPS)
    simp_runge = integrate_with_runge("simpson", f, A, B, eps=EPS)

    print("=== DZ4, variant 7: int_0^1 x e^x dx ===")
    print(f"Точное значение (Ньютон-Лейбниц): {exact:.4f}")
    print(f"max|f''(x)| на [0,1]: {m2:.6f}")
    print()
    print("--- Шаг по второй производной ---")
    print(f"Средние прямоугольники: n={n_mid}, h={h_mid:.6f}, I={i_mid:.6f}, |I-I*|={abs(i_mid-exact):.6e}")
    print(f"Трапеции:              n={n_trap}, h={h_trap:.6f}, I={i_trap:.6f}, |I-I*|={abs(i_trap-exact):.6e}")
    print()
    print("--- Автовыбор шага (правило Рунге) ---")
    print(
        f"Трапеции: n={trap_runge.n}, h={trap_runge.h:.6f}, "
        f"I={trap_runge.value:.6f}, |I-I*|={abs(trap_runge.value-exact):.6e}"
    )
    print(
        f"Симпсон:  n={simp_runge.n}, h={simp_runge.h:.6f}, "
        f"I={simp_runge.value:.6f}, |I-I*|={abs(simp_runge.value-exact):.6e}"
    )


if __name__ == "__main__":
    main()
