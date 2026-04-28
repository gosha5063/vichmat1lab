import numpy as np

from slau_methods import (
    gauss_solve,
    householder_solve,
    residual,
    norms_1_inf,
    condition_number,
)


def variant7_square_systems():
    # 4x4 системы для варианта 7 (из `Данные ДЗ 01 (часть 1)`).
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


def solve_one(A: np.ndarray, b: np.ndarray, x_true: np.ndarray):
    methods = {
        "Гаусс": gauss_solve,
        "Householder(QR)": householder_solve,
    }

    results = []
    for name, solver in methods.items():
        x_hat = solver(A, b)
        r = residual(A, x_hat, b)
        # Невязка
        r1, rinf = norms_1_inf(r)
        # Фактическая погрешность
        e = x_hat - x_true
        e1 = np.linalg.norm(e, ord=1)
        einf = np.linalg.norm(e, ord=np.inf)
        x1_norm = np.linalg.norm(x_true, ord=1)
        xinf_norm = np.linalg.norm(x_true, ord=np.inf)
        rel1_fact = e1 / x1_norm
        relinf_fact = einf / xinf_norm
        # Предельные (априорные) оценки через число обусловленности:
        # ||x - x_hat||/||x|| <= κ(A) * ||r||/||b||
        b1 = np.linalg.norm(b, ord=1)
        binf = np.linalg.norm(b, ord=np.inf)
        k1 = condition_number(A, ord_=1)
        kinf = condition_number(A, ord_=np.inf)
        rel1_bound = k1 * (r1 / b1)
        relinf_bound = kinf * (rinf / binf)
        abs1_bound = x1_norm * rel1_bound
        absinf_bound = xinf_norm * relinf_bound

        results.append(
            {
                "method": name,
                "x_hat": x_hat,
                "r1": r1,
                "rinf": rinf,
                "e1_fact": e1,
                "einf_fact": einf,
                "rel1_fact": rel1_fact,
                "relinf_fact": relinf_fact,
                "abs1_bound": abs1_bound,
                "absinf_bound": absinf_bound,
                "rel1_bound": rel1_bound,
                "relinf_bound": relinf_bound,
            }
        )
    return results


def fmt(x: float) -> str:
    return f"{x:.3e}"


def main():
    systems = variant7_square_systems()

    for A, b, x_true, sys_name in systems:
        print(f"\n=== {sys_name} (Variant 7, 4x4) ===")
        print("x_true =", x_true)

        results = solve_one(A, b, x_true)

        # Таблица: невязка + фактические ошибки + предельные оценки
        print("| Метод | ||r||1 | ||r||inf | ||e||1 фактом | ||e||inf фактом | rel1 фактом | relinf фактом | abs1 предел | absinf предел | rel1 предел | relinf предел |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for res in results:
            print(
                f"| {res['method']} | {fmt(res['r1'])} | {fmt(res['rinf'])} | {fmt(res['e1_fact'])} | {fmt(res['einf_fact'])} | {fmt(res['rel1_fact'])} | {fmt(res['relinf_fact'])} | {fmt(res['abs1_bound'])} | {fmt(res['absinf_bound'])} | {fmt(res['rel1_bound'])} | {fmt(res['relinf_bound'])} |"
            )


if __name__ == "__main__":
    main()

