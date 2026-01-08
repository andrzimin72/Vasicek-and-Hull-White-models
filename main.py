# main.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from shortrate import (
    ShortRateModel, HullWhiteModel, calibrate, bootstrap_calibration,
    swap_rates, libor_rates, vasicek_caplet_price, hull_white_caplet_price,
    hull_white_swaption_price
)

# Market data
LIBOR = np.array([
    [1/12, 1.49078], [2/12, 1.52997], [3/12, 1.60042],
    [6/12, 1.76769], [12/12, 2.04263]
])
SWAP = np.array([
    [2, 2.013], [3, 2.1025], [5, 2.195],
    [7, 2.2585], [10, 2.3457], [15, 2.4447], [30, 2.5055]
])

T_grid = np.linspace(0, 30, 361)
x0_vas = [0.02, 0.8, 0.03, 0.012]  # Better initial guess
x0_hw = [0.0, 0.8, 0.0, 0.012]

# --- Bootstrap Vasicek ---
print("Calibrating Vasicek with bootstrapping...")
mean_params, std_params = bootstrap_calibration('Vasicek', T_grid, LIBOR, SWAP, x0_vas, n_boot=200)
print("Vasicek parameters (mean ± std):")
for name, mu, std in zip(['r0', 'kappa', 'theta', 'sigma'], mean_params, std_params):
    print(f"  {name}: {mu:.4f} ± {std:.4f}")

# --- Create initial curve with cubic spline ---
def create_initial_curve(libor, swap):
    maturities = np.concatenate([libor[:, 0], swap[:, 0]])
    rates = np.concatenate([libor[:, 1]/100, swap[:, 1]/100])
    cs = CubicSpline(maturities, rates, bc_type='natural')
    def df_func(T):
        T = np.asarray(T)
        df = np.ones_like(T, dtype=float)
        mask = T > 0
        if np.any(mask):
            r = cs(T[mask])
            df[mask] = np.exp(-r * T[mask])
        return df
    return df_func

P0T_func = create_initial_curve(LIBOR, SWAP)
P0_initial = P0T_func(T_grid)

# --- Calibrate Hull-White ---
print("\nCalibrating Hull-White...")
hw_params, p_hw, L_hw, S_hw = calibrate('HullWhite', T_grid, LIBOR, SWAP, x0_hw, P0T_func)

# Enforce minimum volatility
MIN_SIGMA = 0.005
if hw_params[3] < MIN_SIGMA:
    print(f"⚠️  Sigma too low ({hw_params[3]:.4f}), setting to {MIN_SIGMA:.3%}")
    hw_params[3] = MIN_SIGMA
    p_hw = HullWhiteModel(hw_params[1], hw_params[3], P0T_func, T_grid).zcb_price(hw_params[0])
    L_hw = libor_rates(T_grid, p_hw, LIBOR[:, 0])
    S_hw = swap_rates(T_grid, p_hw, SWAP[:, 0])
print(f"Hull-White: kappa={hw_params[1]:.4f}, sigma={hw_params[3]:.4f}")

# --- Generate Vasicek curve ---
p_vas = ShortRateModel(mean_params[0], mean_params[1], mean_params[2], mean_params[3], T_grid, 'Vasicek').zcb_price()

# --- Plot LIBOR/Swap Fit ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(LIBOR[:, 0], LIBOR[:, 1], c='black', label='Market')
plt.plot(LIBOR[:, 0], libor_rates(T_grid, p_vas, LIBOR[:, 0]), 'o--', label='Vasicek')
plt.plot(LIBOR[:, 0], L_hw[:len(LIBOR)], 's:', label='Hull-White')
plt.xlabel('Maturity (Y)'); plt.ylabel('Rate (%)'); plt.title('LIBOR'); plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(SWAP[:, 0], SWAP[:, 1], c='black', label='Market')
plt.plot(SWAP[:, 0], swap_rates(T_grid, p_vas, SWAP[:, 0]), 'o--', label='Vasicek')
plt.plot(SWAP[:, 0], S_hw, 's:', label='Hull-White')
plt.xlabel('Maturity (Y)'); plt.ylabel('Rate (%)'); plt.title('Swap'); plt.legend()
plt.tight_layout()
plt.show()

# --- Plot ZCB Curves ---
plt.figure(figsize=(10, 6))
plt.plot(T_grid, P0_initial, 'k-', linewidth=2, label='Initial Market')
plt.plot(T_grid, p_vas, 'b--', label='Vasicek')
plt.plot(T_grid, p_hw, 'r:', label='Hull-White')
plt.xlabel('Maturity (Years)'); plt.ylabel('ZCB Price')
plt.title('Zero-Coupon Bond Curves'); plt.legend(); plt.grid(True)
plt.show()

# --- Export Yield Curves ---
def spot_rate(P, T):
    T = np.maximum(T, 1e-8)
    return -np.log(P) / T

def forward_rate(P, T, dt=1e-4):
    T = np.maximum(T, dt)
    P_up = np.interp(T + dt, T, P, left=1.0, right=P[-1])
    P_dn = np.interp(T - dt, T, P, left=1.0, right=P[-1])
    return -(np.log(P_up) - np.log(P_dn)) / (2 * dt)

df_curves = pd.DataFrame({
    'Maturity': T_grid,
    'Initial_ZCB': P0_initial,
    'Vasicek_ZCB': p_vas,
    'HullWhite_ZCB': p_hw,
    'Initial_Spot': spot_rate(P0_initial, T_grid),
    'Vasicek_Spot': spot_rate(p_vas, T_grid),
    'HullWhite_Spot': spot_rate(p_hw, T_grid),
    'Initial_Forward': forward_rate(P0_initial, T_grid),
    'Vasicek_Forward': forward_rate(p_vas, T_grid),
    'HullWhite_Forward': forward_rate(p_hw, T_grid)
})
df_curves.to_csv('yield_curves.csv', index=False)
print("\n✅ Yield curves saved to 'yield_curves.csv'")

# --- Caplet Breakdown ---
cap_maturity = 5.0
K_strike = 0.025
freq = 4
accruals = np.arange(1/freq, cap_maturity + 1/freq, 1/freq)

vas_caplets = [
    vasicek_caplet_price(mean_params[0], mean_params[1], mean_params[2], mean_params[3], T1, T2, K_strike)
    for T1, T2 in zip(accruals[:-1], accruals[1:])
]
hw_caplets = [
    hull_white_caplet_price(hw_params[1], hw_params[3], P0T_func, T1, T2, K_strike, hw_params[0])
    for T1, T2 in zip(accruals[:-1], accruals[1:])
]

plt.figure(figsize=(10, 5))
caplet_times = accruals[1:]
plt.bar(caplet_times - 0.05, vas_caplets, width=0.1, label='Vasicek', alpha=0.8)
plt.bar(caplet_times + 0.05, hw_caplets, width=0.1, label='Hull-White', alpha=0.8)
plt.xlabel('Caplet Expiry (Years)'); plt.ylabel('Price')
plt.title(f'Caplet Breakdown (5Y Cap, K={K_strike:.1%})'); plt.legend(); plt.grid(True, axis='y')
plt.show()

print(f"\n5Y Cap Total Prices (K={K_strike:.1%}):")
print(f"  Vasicek: {sum(vas_caplets):.6f}")
print(f"  Hull-White: {sum(hw_caplets):.6f}")

# --- Swaption Pricing ---
swaption_maturity, swaption_tenor, swaption_freq, swaption_K = 5.0, 10.0, 2, 0.025
hw_swaption = hull_white_swaption_price(
    hw_params[1], hw_params[3], P0T_func,
    swaption_maturity, swaption_tenor, swaption_freq, swaption_K, hw_params[0]
)

print(f"\n{swaption_maturity}Yx{swaption_tenor}Y Swaption (K={swaption_K:.1%}):")
print(f"  Hull-White: {hw_swaption:.6f}")

# --- Save Parameters ---
df_params = pd.DataFrame({
    'Model': ['Vasicek', 'HullWhite'],
    'r0': [mean_params[0], hw_params[0]],
    'kappa': [mean_params[1], hw_params[1]],
    'theta': [mean_params[2], np.nan],
    'sigma': [mean_params[3], hw_params[3]],
    'r0_std': [std_params[0], np.nan],
    'kappa_std': [std_params[1], np.nan],
    'theta_std': [std_params[2], np.nan],
    'sigma_std': [std_params[3], np.nan]
})
df_params.to_csv('calibrated_parameters.csv', index=False)
print("✅ Parameters saved to 'calibrated_parameters.csv'")