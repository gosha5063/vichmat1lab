import numpy as np

from slau_methods import gauss_solve, householder_solve, residual, norms_1_inf


def variant7_square_good():
    # Из `Данные ДЗ 01 (часть 1)` для варианта 7: 4x4 система
    A = np.array(
        [
            [127.8, 8.03, 1.4, -2.36],
            [0.27, 136.4, -0.16, -4.55],
            [-3.84, 5.37, -111.0, 1.56],
            [-6.53, 6.72, 2.88, 47.2],
        ],
        dtype=float,
    )
    b = np.array([-1008.64, 516.62, 394.56, 353.68], dtype=float)
    x_true = np.array([-8.0, 4.0, -3.0, 6.0], dtype=float)
    return A, b, x_true


def main():
    A, b, x_true = variant7_square_good()

    x_g = gauss_solve(A, b)
    x_h = householder_solve(A, b)

    r_g = residual(A, x_g, b)
    r_h = residual(A, x_h, b)

    print("Variant 7, good system (4x4)")
    print("x_true      =", x_true)
    print("x_gauss     =", np.round(x_g, 10))
    print("x_household =", np.round(x_h, 10))

    ng1, nginf = norms_1_inf(r_g)
    nh1, nhinf = norms_1_inf(r_h)
    print(f"residual norms (Gauss):       ||r||1={ng1:.3e}, ||r||inf={nginf:.3e}")
    print(f"residual norms (Householder): ||r||1={nh1:.3e}, ||r||inf={nhinf:.3e}")


if __name__ == "__main__":
    main()

