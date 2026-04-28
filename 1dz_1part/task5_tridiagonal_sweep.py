import numpy as np

from slau_methods import sweep_tridiagonal, build_tridiagonal_matrix, residual, norms_1_inf


def variant7_tridiagonal():
    # Размер n=6 (из `Данные ДЗ 01 (часть 2)`, вариант 7):
    # a = (a2..an), b = (b1..bn), c = (c1..c_{n-1}), d = (d1..dn)
    a_sub = np.array([2, 1, 0, 1, 1], dtype=float)
    b_diag = np.array([141, 98, 74, 87, -169, 51], dtype=float)
    c_sup = np.array([1, 0, 1, 0, 1], dtype=float)
    d = np.array([14, 10, 7, 9, -17, 5], dtype=float)
    return a_sub, b_diag, c_sup, d


def main():
    a_sub, b_diag, c_sup, d = variant7_tridiagonal()
    A = build_tridiagonal_matrix(a_sub, b_diag, c_sup)

    x = sweep_tridiagonal(a_sub, b_diag, c_sup, d)
    r = residual(A, x, d)
    r1, rinf = norms_1_inf(r)

    print("Variant 7, tridiagonal system (n=6)")
    print("x_sweep =", np.round(x, 10))
    print(f"residual norms: ||r||1={r1:.3e}, ||r||inf={rinf:.3e}")


if __name__ == "__main__":
    main()

