import numpy as np

from slau_methods import (
    gauss_inverse,
    householder_inverse,
    norms_1_inf,
)


def variant7_square_systems():
    A_good = np.array(
        [
            [127.8, 8.03, 1.4, -2.36],
            [0.27, 136.4, -0.16, -4.55],
            [-3.84, 5.37, -111.0, 1.56],
            [-6.53, 6.72, 2.88, 47.2],
        ],
        dtype=float,
    )
    b_good = np.array([-1008.64, 516.62, 394.56, 353.68], dtype=float)
    x_good = np.array([-8.0, 4.0, -3.0, 6.0], dtype=float)

    A_bad = np.array(
        [
            [3.897, -3.894, 19.062, 27.258],
            [29.16, -29.157, 158.952, 198.716],
            [0.972, -0.972, 4.761, 6.804],
            [2.916, -2.916, 16.59, 19.643],
        ],
        dtype=float,
    )
    b_bad = np.array([-17.205, -426.238, -4.347, -55.336], dtype=float)
    x_bad = np.array([13.0, 14.0, -15.0, 10.0], dtype=float)

    return (A_good, x_good, "Хорошо обусловленная"), (A_bad, x_bad, "Плохо обусловленная")


def check_inverse(A: np.ndarray, A_inv: np.ndarray):
    n = A.shape[0]
    I = np.eye(n, dtype=float)
    M = A_inv @ A - I
    n1, ninf = norms_1_inf(M)
    return n1, ninf


def main():
    systems = variant7_square_systems()

    for A, _, sys_name in systems:
        n = A.shape[0]
        print(f"\n=== {sys_name} (Variant 7, 4x4) ===")
        print(f"n={n}")

        A_inv_g = gauss_inverse(A)
        A_inv_h = householder_inverse(A)

        e1_g, einf_g = check_inverse(A, A_inv_g)
        e1_h, einf_h = check_inverse(A, A_inv_h)

        print("| Метод | ||A^{-1}A - E||1 | ||A^{-1}A - E||inf |")
        print("|---|---:|---:|")
        print(f"| Гаусс | {e1_g:.3e} | {einf_g:.3e} |")
        print(f"| Householder(QR) | {e1_h:.3e} | {einf_h:.3e} |")


if __name__ == "__main__":
    main()

