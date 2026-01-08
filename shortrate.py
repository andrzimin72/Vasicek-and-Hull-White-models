# shortrate.py
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Tuple, Optional, Callable

class ShortRateModel:
    def __init__(self, r0: float, kappa: float, theta: float, sigma: float, tau, model: str):
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.tau = np.asarray(tau)
        self.model = model

    def zcb_price(self) -> np.ndarray:
        if self.model == 'Vasicek':
            B = (1 - np.exp(-self.kappa * self.tau)) / self.kappa
            A = ((self.theta - self.sigma**2 / (2 * self.kappa**2)) * (B - self.tau)
                 - self.sigma**2 * B**2 / (4 * self.kappa))
        elif self.model == 'CIR':
            g = np.sqrt(self.kappa**2 + 2 * self.sigma**2)
            tmp = 2 * self.kappa * self.theta / self.sigma**2
            A = tmp * np.log(
                np.exp(self.kappa * self.tau / 2) /
                (np.cosh(g * self.tau / 2) + (self.kappa / g) * np.sinh(g * self.tau / 2))
            )
            B = (2 * np.tanh(g * self.tau / 2)) / (self.kappa * np.tanh(g * self.tau / 2) + g)
        else:
            raise ValueError("Model must be 'Vasicek' or 'CIR'")
        return np.exp(A - B * self.r0)


class HullWhiteModel:
    def __init__(self, kappa: float, sigma: float, P0T_func: Callable, tau):
        self.kappa = kappa
        self.sigma = sigma
        self.P0T_func = P0T_func
        self.tau = np.asarray(tau)

    def zcb_price(self, r0: float = 0.0) -> np.ndarray:
        T = self.tau
        B = (1 - np.exp(-self.kappa * T)) / self.kappa
        # Compute f(0,0) from initial curve using central difference
        dt = 1e-5
        T_safe = np.maximum(T, dt)
        logP_up = np.log(self.P0T_func(T_safe + dt))
        logP_dn = np.log(self.P0T_func(np.maximum(T_safe - dt, 0)))
        f0 = -(logP_up - logP_dn) / (2 * dt)
        f0_0 = f0[0] if len(f0) > 0 else 0.0
        A = np.log(self.P0T_func(T)) + B * f0_0 - (self.sigma**2 / (4 * self.kappa)) * B**2 * (1 - np.exp(-2 * self.kappa * T))
        return np.exp(A - B * r0)


def swap_rates(tau: np.ndarray, p: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if len(tau) == 0 or len(mat) == 0:
        return np.array([])
    tmax = mat[-1]
    t_grid = np.arange(0.5, tmax + 0.5, 0.5)
    p_interp = np.interp(t_grid, tau, p, left=1.0, right=p[-1])
    disc_sum = np.cumsum(p_interp)
    p_mat = np.interp(mat, tau, p, left=1.0, right=p[-1])
    idx = np.minimum((2 * mat).astype(int) - 1, len(disc_sum) - 1)
    return 100 * 2 * (1 - p_mat) / disc_sum[idx]


def libor_rates(tau: np.ndarray, p: np.ndarray, mat: np.ndarray) -> np.ndarray:
    p_mat = np.interp(mat, tau, p, left=1.0, right=p[-1])
    mat = np.maximum(mat, 1e-8)
    return 100 * (1 / p_mat - 1) / mat


def vasicek_caplet_price(r0, kappa, theta, sigma, T1, T2, K):
    if T1 <= 0 or T2 <= T1:
        return 0.0
    delta = T2 - T1
    B = lambda t, T: (1 - np.exp(-kappa * (T - t))) / kappa
    A = lambda t, T: ((theta - sigma**2 / (2 * kappa**2)) * (B(t, T) - (T - t))
                      - sigma**2 * B(t, T)**2 / (4 * kappa))
    P0T1 = np.exp(A(0, T1) - B(0, T1) * r0)
    P0T2 = np.exp(A(0, T2) - B(0, T2) * r0)
    F = (1 / delta) * (P0T1 / P0T2 - 1)
    if F <= 0:
        return 0.0
    v = (sigma / kappa) * (1 - np.exp(-kappa * T1)) * B(T1, T2)
    if v <= 1e-12:
        return max(F - K, 0) * P0T2 * delta
    d1 = (np.log(F / K) + 0.5 * v**2 * T1) / (v * np.sqrt(T1))
    d2 = d1 - v * np.sqrt(T1)
    return delta * P0T2 * (F * norm.cdf(d1) - K * norm.cdf(d2))


def vasicek_cap_price(r0, kappa, theta, sigma, cap_maturity, K, freq=4):
    if cap_maturity <= 0:
        return 0.0
    accruals = np.arange(1/freq, cap_maturity + 1/freq, 1/freq)
    total = 0.0
    for i in range(1, len(accruals)):
        total += vasicek_caplet_price(r0, kappa, theta, sigma, accruals[i-1], accruals[i], K)
    return total


def hull_white_caplet_price(kappa, sigma, P0T_func, T1, T2, K, r0=0.0):
    if T1 <= 0 or T2 <= T1:
        return 0.0
    delta = T2 - T1
    P0T1 = P0T_func(T1)
    P0T2 = P0T_func(T2)
    F = (1 / delta) * (P0T1 / P0T2 - 1)
    if F <= 0:
        return 0.0
    B_T1_T2 = (1 - np.exp(-kappa * (T2 - T1))) / kappa
    v_sq = (sigma**2 / kappa**2) * B_T1_T2**2 * (1 - np.exp(-2 * kappa * T1))
    v = np.sqrt(v_sq)
    if v <= 1e-12:
        return max(F - K, 0) * P0T2 * delta
    d1 = (np.log(F / K) + 0.5 * v_sq * T1) / (v * np.sqrt(T1))
    d2 = d1 - v * np.sqrt(T1)
    return delta * P0T2 * (F * norm.cdf(d1) - K * norm.cdf(d2))


def hull_white_cap_price(kappa, sigma, P0T_func, cap_maturity, K, freq=4, r0=0.0):
    if cap_maturity <= 0:
        return 0.0
    accruals = np.arange(1/freq, cap_maturity + 1/freq, 1/freq)
    total = 0.0
    for i in range(1, len(accruals)):
        total += hull_white_caplet_price(kappa, sigma, P0T_func, accruals[i-1], accruals[i], K, r0)
    return total


def hull_white_swaption_price(kappa, sigma, P0T_func, option_maturity, swap_tenor, swap_freq, K, r0=0.0):
    if option_maturity <= 0 or swap_tenor <= 0:
        return 0.0
    accruals = np.arange(option_maturity, option_maturity + swap_tenor + 1e-9, 1/swap_freq)
    T0 = option_maturity
    Tn = accruals[-1]
    deltas = np.diff(accruals)
    P0Ti = np.array([P0T_func(T) for T in accruals[1:]])
    annuity = np.sum(deltas * P0Ti)
    if annuity <= 0:
        return 0.0
    S0 = (P0T_func(T0) - P0Ti[-1]) / annuity
    def B(t, T): return (1 - np.exp(-kappa * (T - t))) / kappa
    v_s_sq = 0.0
    for i, Ti in enumerate(accruals[1:]):
        wi = deltas[i] * P0Ti[i] / annuity
        vol_i = sigma * B(T0, Ti) / kappa
        v_s_sq += wi * vol_i**2 * (1 - np.exp(-2 * kappa * T0))
    v_s = np.sqrt(v_s_sq)
    if v_s < 1e-12:
        return max(S0 - K, 0) * annuity
    d1 = (np.log(S0 / K) + 0.5 * v_s**2 * T0) / (v_s * np.sqrt(T0))
    d2 = d1 - v_s * np.sqrt(T0)
    return annuity * (S0 * norm.cdf(d1) - K * norm.cdf(d2))


def _objective(params, tau, libor, swap, model, P0T_func=None):
    r0, kappa, theta, sigma = params
    # Enforce realistic bounds
    if not (-0.1 <= r0 <= 0.2 and 0.01 <= kappa <= 10.0 and 0.0 <= theta <= 0.2 and 0.001 <= sigma <= 0.1):
        return 1e6
    try:
        if model == 'HullWhite':
            p = HullWhiteModel(kappa, sigma, P0T_func, tau).zcb_price(r0)
        else:
            p = ShortRateModel(r0, kappa, theta, sigma, tau, model).zcb_price()
        S = swap_rates(tau, p, swap[:, 0])
        L = libor_rates(tau, p, libor[:, 0])
        # Use absolute error to avoid instability
        err = np.sum(np.abs(S - swap[:, 1])) + np.sum(np.abs(L - libor[:, 1]))
        return err
    except Exception:
        return 1e6


def calibrate(model: str, tau: np.ndarray, libor: np.ndarray, swap: np.ndarray,
              x0: np.ndarray, P0T_func: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bounds = [(-0.1, 0.2), (0.01, 10.0), (0.0, 0.2), (0.001, 0.1)]
    res = minimize(
        _objective, x0,
        args=(tau, libor, swap, model, P0T_func),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'disp': False}
    )
    r0, kappa, theta, sigma = res.x
    if model == 'HullWhite':
        p = HullWhiteModel(kappa, sigma, P0T_func, tau).zcb_price(r0)
    else:
        p = ShortRateModel(r0, kappa, theta, sigma, tau, model).zcb_price()
    L = libor_rates(tau, p, libor[:, 0])
    S = swap_rates(tau, p, swap[:, 0])
    return res.x, p, L, S


def bootstrap_calibration(model: str, tau: np.ndarray, libor: np.ndarray, swap: np.ndarray,
                         x0: np.ndarray, n_boot: int = 100, P0T_func: Optional[Callable] = None,
                         seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    all_params = []
    n_libor, n_swap = len(libor), len(swap)
    for _ in range(n_boot):
        libor_boot = libor[np.random.choice(n_libor, n_libor, replace=True)]
        swap_boot = swap[np.random.choice(n_swap, n_swap, replace=True)]
        try:
            params, _, _, _ = calibrate(model, tau, libor_boot, swap_boot, x0, P0T_func)
            all_params.append(params)
        except Exception:
            continue
    if len(all_params) == 0:
        raise RuntimeError("Bootstrap failed: no valid calibrations")
    all_params = np.array(all_params)
    return np.mean(all_params, axis=0), np.std(all_params, axis=0)