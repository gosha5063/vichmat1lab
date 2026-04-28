import numpy as np

from slau_methods import (
    gauss_solve,
    householder_solve,
    residual,
    norms_1_inf,
    condition_number,
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

    return (A_good, b_good, x_good, "Хорошо обусловленная"), (A_bad, b_bad, x_bad, "Плохо обусловленная")


def order10(kappa: float) -> int:
    return int(np.floor(np.log10(kappa))) if kappa > 0 else 0


def main():
    systems = variant7_square_systems()

    for A, b, x_true, sys_name in systems:
        k1 = condition_number(A, ord_=1)
        kinf = condition_number(A, ord_=np.inf)
        print(f"\n=== {sys_name} (Variant 7, 4x4) ===")
        print(f"cond_1(A)   = {k1:.6e}, ~10^{order10(k1)}")
        print(f"cond_inf(A) = {kinf:.6e}, ~10^{order10(kinf)}")

        methods = {"Гаусс": gauss_solve, "Householder(QR)": householder_solve}
        x1_norm = np.linalg.norm(x_true, ord=1)
        xinf_norm = np.linalg.norm(x_true, ord=np.inf)
        b1 = np.linalg.norm(b, ord=1)
        binf = np.linalg.norm(b, ord=np.inf)

        print("| Метод | ||r||1 | ||r||inf | ||e||1 факт | ||e||inf факт | abs1 предел | absinf предел |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, solver in methods.items():
            x_hat = solver(A, b)
            r = residual(A, x_hat, b)
            r1, rinf = norms_1_inf(r)
            e = x_hat - x_true
            e1 = np.linalg.norm(e, ord=1)
            einf = np.linalg.norm(e, ord=np.inf)

            # Предельные оценки через κ(A)
            rel1_bound = k1 * (r1 / b1)
            relinf_bound = kinf * (rinf / binf)
            abs1_bound = x1_norm * rel1_bound
            absinf_bound = xinf_norm * relinf_bound

            print(
                f"| {name} | {r1:.3e} | {rinf:.3e} | {e1:.3e} | {einf:.3e} | {abs1_bound:.3e} | {absinf_bound:.3e} |"
            )


if __name__ == "__main__":
    main()

