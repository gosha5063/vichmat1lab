import numpy as np


def gauss_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Решение Ax=B методом Гаусса с выбором главного элемента по столбцу.

    A: (n,n)
    B: (n,) или (n,m)
    """
    A = np.array(A, dtype=float, copy=True)
    B = np.array(B, dtype=float, copy=True)

    if B.ndim == 1:
        B = B.reshape(-1, 1)

    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if B.shape[0] != n:
        raise ValueError("B rows must match A")

    # Прямой ход: приведение к верхнетреугольному виду
    for k in range(n - 1):
        pivot = k + np.argmax(np.abs(A[k:, k]))
        if np.isclose(A[pivot, k], 0.0):
            raise np.linalg.LinAlgError("Matrix is singular (zero pivot)")
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            B[[k, pivot]] = B[[pivot, k]]

        factor = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k:] -= factor[:, None] * A[k, k:]
        B[k + 1 :, :] -= factor[:, None] * B[k, :]

    # Обратный ход
    X = np.zeros_like(B)
    for i in range(n - 1, -1, -1):
        if np.isclose(A[i, i], 0.0):
            raise np.linalg.LinAlgError("Matrix is singular (zero diagonal)")
        X[i, :] = (B[i, :] - A[i, i + 1 :] @ X[i + 1 :, :]) / A[i, i]

    return X.ravel() if X.shape[1] == 1 else X


def gauss_inverse(A: np.ndarray) -> np.ndarray:
    """Находим A^{-1} как решение A X = I (столбцы по очереди)."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    I = np.eye(n, dtype=float)
    return gauss_solve(A, I)


def back_substitution_upper_tri(R: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Решает верхнетреугольную систему R X = Y."""
    R = np.asarray(R, dtype=float)
    Y = np.asarray(Y, dtype=float)

    n = R.shape[0]
    if Y.ndim == 1:
        Y = Y.reshape(n, 1)

    X = np.zeros_like(Y)
    for i in range(n - 1, -1, -1):
        X[i, :] = (Y[i, :] - R[i, i + 1 :] @ X[i + 1 :, :]) / R[i, i]
    return X.ravel() if X.shape[1] == 1 else X


def householder_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Решение Ax=B через QR-разложение с отражениями Хаусхолдера.
    Формируем R (Q явно не строим) и применяем те же отражения к B.
    """
    R = np.array(A, dtype=float, copy=True)
    Y = np.array(B, dtype=float, copy=True)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n = R.shape[0]
    if R.shape != (n, n):
        raise ValueError("A must be square")
    if Y.shape[0] != n:
        raise ValueError("B rows must match A")

    for k in range(n - 1):
        x = R[k:, k]
        normx = np.linalg.norm(x)
        if np.isclose(normx, 0.0):
            continue

        sign = 1.0 if x[0] >= 0 else -1.0
        alpha = -sign * normx

        v = x.copy()
        v[0] -= alpha
        vv = v @ v
        if np.isclose(vv, 0.0):
            continue
        beta = 2.0 / vv

        # Применяем отражение к R: R <- H R, где H = I - beta v v^T
        R[k:, k:] -= beta * np.outer(v, v @ R[k:, k:])
        # И к Y: Y <- H Y
        Y[k:, :] -= beta * np.outer(v, v @ Y[k:, :])

    # Теперь R верхнетреугольная, решаем R X = Y
    X = back_substitution_upper_tri(R, Y)
    return X


def householder_inverse(A: np.ndarray) -> np.ndarray:
    """Находим A^{-1} как решение A X = I (QR/Хаусхолдер)."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    I = np.eye(n, dtype=float)
    return householder_solve(A, I)


def residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Невязка r = b - A x."""
    return np.asarray(b, dtype=float) - np.asarray(A, dtype=float) @ np.asarray(x, dtype=float)


def norms_1_inf(v: np.ndarray) -> tuple[float, float]:
    """(||v||_1, ||v||_inf)."""
    v = np.asarray(v, dtype=float)
    return float(np.linalg.norm(v, ord=1)), float(np.linalg.norm(v, ord=np.inf))


def condition_number(A: np.ndarray, ord_: int) -> float:
    """Число обусловленности κ_A( ord_ )."""
    return float(np.linalg.cond(np.asarray(A, dtype=float), p=ord_))


def sweep_tridiagonal(a_sub: np.ndarray, b_diag: np.ndarray, c_sup: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Прогонка для СЛАУ с трехдиагональной матрицей размера n.

    a_sub: (n-1,) элементы поддиагонали a2..an
    b_diag: (n,)   диагональ b1..bn
    c_sup: (n-1,)  наддиагональ c1..c_{n-1}
    d: (n,) правые части
    """
    a_sub = np.asarray(a_sub, dtype=float)
    b_diag = np.asarray(b_diag, dtype=float)
    c_sup = np.asarray(c_sup, dtype=float)
    d = np.asarray(d, dtype=float)

    n = b_diag.size
    if a_sub.size != n - 1 or c_sup.size != n - 1 or d.size != n:
        raise ValueError("Invalid tridiagonal sizes")

    a = np.zeros(n, dtype=float)
    c = np.zeros(n, dtype=float)
    a[1:] = a_sub
    c[:-1] = c_sup

    alpha = np.zeros(n, dtype=float)
    beta = np.zeros(n, dtype=float)

    # i=0
    if np.isclose(b_diag[0], 0.0):
        raise np.linalg.LinAlgError("Zero on diagonal at i=1")
    alpha[0] = -c[0] / b_diag[0]
    beta[0] = d[0] / b_diag[0]

    # i=1..n-2
    for i in range(1, n - 1):
        den = b_diag[i] + a[i] * alpha[i - 1]
        if np.isclose(den, 0.0):
            raise np.linalg.LinAlgError("Zero denominator during sweep")
        alpha[i] = -c[i] / den
        beta[i] = (d[i] - a[i] * beta[i - 1]) / den

    # i=n-1
    den = b_diag[n - 1] + a[n - 1] * alpha[n - 2]
    if np.isclose(den, 0.0):
        raise np.linalg.LinAlgError("Zero denominator during sweep at last step")
    beta[n - 1] = (d[n - 1] - a[n - 1] * beta[n - 2]) / den

    # Обратный ход
    x = np.zeros(n, dtype=float)
    x[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    return x


def build_tridiagonal_matrix(a_sub: np.ndarray, b_diag: np.ndarray, c_sup: np.ndarray) -> np.ndarray:
    """Только для валидации/вычисления невязки."""
    a_sub = np.asarray(a_sub, dtype=float)
    b_diag = np.asarray(b_diag, dtype=float)
    c_sup = np.asarray(c_sup, dtype=float)
    n = b_diag.size
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = b_diag[i]
        if i > 0:
            A[i, i - 1] = a_sub[i - 1]
        if i < n - 1:
            A[i, i + 1] = c_sup[i]
    return A

