#!/usr/bin/env python3
"""
Global Rank Envelope Diagnostic Framework for Linear Regression.

Implements the five-stage pipeline:
  I.   Data preparation (80/20 train-test split)
  II.  OLS model fitting & empirical diagnostic curves (LOESS-based)
  III. Simulation-based global rank envelope construction (B=1000)
  IV.  Formal hypothesis testing via global p-values
  V.   Integrated reporting (report.md + diagnostic plots)

Reference: Myllymäki et al. (2017) Global envelope tests for spatial processes.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_data.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
IMAGES_DIR = os.path.join(BASE_DIR, "report", "images")
REPORT_PATH = os.path.join(BASE_DIR, "report", "report.md")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------- constants ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2
B = 1000            # number of simulations
M = 200             # evaluation grid points
ALPHA = 0.05        # significance level
LOESS_SPAN = 0.5    # LOESS smoothing span
LOESS_DEGREE = 2    # LOESS local polynomial degree

np.random.seed(RANDOM_STATE)


# ============================================================
#  LOESS (locally estimated scatterplot smoothing) — manual impl
# ============================================================
def _tricube_weight(d, span):
    """Tricube weight function for LOESS."""
    max_d = np.max(d) if np.max(d) > 0 else 1.0
    u = d / max_d
    w = np.where(np.abs(u) < 1, (1 - np.abs(u) ** 3) ** 3, 0.0)
    return w


def loess_smooth(x, y, x_grid, span=LOESS_SPAN, degree=LOESS_DEGREE):
    """
    LOESS regression evaluated at x_grid points.

    Parameters
    ----------
    x : (n,) array — predictor values
    y : (n,) array — response values
    x_grid : (m,) array — evaluation grid
    span : float — fraction of points used in each local neighbourhood
    degree : int — polynomial degree (1 or 2)

    Returns
    -------
    y_smooth : (m,) array — LOESS fit at grid points
    """
    n = len(x)
    k = max(int(np.ceil(n * span)), degree + 2)  # ensure at least degree+2 neighbours

    y_smooth = np.full(len(x_grid), np.nan)
    for i, x0 in enumerate(x_grid):
        # distances from x0 to all observations
        d = np.abs(x - x0)
        idx = np.argsort(d)
        d_sorted = d[idx]
        # nearest k neighbours
        d_k = d_sorted[:k]
        y_k = y[idx[:k]]
        x_k = x[idx[:k]]

        # tricube weights scaled by the maximum distance within the neighbourhood
        max_d_k = d_k[-1] if d_k[-1] > 0 else 1.0
        u = d_k / max_d_k
        w = np.where(np.abs(u) < 1, (1 - np.abs(u) ** 3) ** 3, 0.0)

        # weighted least squares polynomial fit
        if degree == 1:
            A = np.column_stack([np.ones_like(x_k), x_k - x0])
        else:  # degree == 2
            A = np.column_stack([np.ones_like(x_k), x_k - x0, (x_k - x0) ** 2])

        W = np.diag(w)
        try:
            beta = np.linalg.lstsq(A.T @ W @ A, A.T @ (w * y_k), rcond=None)[0]
            y_smooth[i] = beta[0]
        except np.linalg.LinAlgError:
            y_smooth[i] = np.mean(y_k)

    return y_smooth


# ============================================================
#  I. Data Preparation
# ============================================================
def load_and_split_data():
    """Load synthetic data and perform 80/20 train-test split."""
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, df.columns.tolist()


# ============================================================
#  II. Model Fitting & Empirical Diagnostic Curves
# ============================================================
def fit_ols(X_train, y_train):
    """
    Fit OLS model. Returns coefficients, fitted values, residuals,
    standardized residuals, sigma_hat, leverages, and model R².
    """
    n, p = X_train.shape
    X_with_intercept = np.column_stack([np.ones(n), X_train])

    beta_hat = np.linalg.lstsq(X_with_intercept, y_train, rcond=None)[0]
    y_hat = X_with_intercept @ beta_hat
    residuals = y_train - y_hat

    # sigma^2 estimate (unbiased)
    sigma2 = np.sum(residuals ** 2) / (n - p - 1)
    sigma_hat = np.sqrt(sigma2)

    # leverages h_ii = diag(X (X^T X)^{-1} X^T)
    try:
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        H = X_with_intercept @ XtX_inv @ X_with_intercept.T
        h_ii = np.diag(H)
    except np.linalg.LinAlgError:
        h_ii = np.full(n, (p + 1) / n)

    # standardized residuals
    standardized_residuals = residuals / (sigma_hat * np.sqrt(1.0 - h_ii + 1e-10))

    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    r2_train = 1.0 - ss_res / ss_tot

    # F-statistic
    f_statistic = (ss_tot - ss_res) / p / (ss_res / (n - p - 1))
    f_pval = 1.0 - stats.f.cdf(f_statistic, p, n - p - 1)

    return {
        "beta_hat": beta_hat,
        "intercept": beta_hat[0],
        "coef": beta_hat[1:],
        "y_hat": y_hat,
        "residuals": residuals,
        "standardized_residuals": standardized_residuals,
        "sigma_hat": sigma_hat,
        "sigma2": sigma2,
        "h_ii": h_ii,
        "r2_train": r2_train,
        "f_statistic": f_statistic,
        "f_pval": f_pval,
    }


def compute_observed_statistics(fit_result, grid_f):
    """
    Compute the three observed functional diagnostic curves.

    Returns
    -------
    dict with keys: 'rvf' (residuals-vs-fitted), 'qq' (Q-Q difference),
                    'sl' (scale-location)
    """
    n_obs = len(fit_result["residuals"])

    # a) T_RVF — LOESS of residuals vs fitted values
    y_hat = fit_result["y_hat"]
    residuals = fit_result["residuals"]
    T_rvf_obs = loess_smooth(y_hat, residuals, grid_f, span=LOESS_SPAN)

    # b) Δ_QQ — ordered standardized residuals vs theoretical quantiles
    std_res = fit_result["standardized_residuals"]
    d_ordered = np.sort(std_res)
    q_theoretical = stats.norm.ppf((np.arange(1, n_obs + 1) - 0.5) / n_obs)
    # interpolate to m = 200 points
    q_theo_grid = stats.norm.ppf((np.arange(1, M + 1) - 0.5) / M)
    d_interp = np.interp(q_theo_grid, q_theoretical, d_ordered)
    delta_qq_obs = d_interp - q_theo_grid

    # c) T_SL — LOESS of sqrt(|standardized residuals|) vs fitted values
    sqrt_abs_std = np.sqrt(np.abs(fit_result["standardized_residuals"]))
    T_sl_obs = loess_smooth(y_hat, sqrt_abs_std, grid_f, span=LOESS_SPAN)

    return {
        "rvf": T_rvf_obs,
        "qq_delta": delta_qq_obs,
        "sl": T_sl_obs,
        "q_theoretical": q_theo_grid,
        "grid_f": grid_f,
        "d_interp": d_interp,
    }


# ============================================================
#  III. Simulation & Global Rank Envelope Construction
# ============================================================
def simulate_null_statistics(X_train, fit_result, grid_f):
    """
    Simulate B response vectors under H0 and recompute the three
    functional statistics for each.

    Returns
    -------
    dict with keys: 'rvf_sim', 'qq_sim', 'sl_sim' — each a (B, M) array.
    Also returns 'q_theoretical' for reference.
    """
    n, p = X_train.shape
    X_with_intercept = np.column_stack([np.ones(n), X_train])
    beta_hat = fit_result["beta_hat"]
    sigma_hat = fit_result["sigma_hat"]
    n_obs = n

    q_theo_grid = stats.norm.ppf((np.arange(1, M + 1) - 0.5) / M)

    # Precompute the theoretical quantiles for the full sample size
    q_theoretical_full = stats.norm.ppf((np.arange(1, n_obs + 1) - 0.5) / n_obs)

    T_rvf_sim = np.zeros((B, M))
    T_qq_sim = np.zeros((B, M))
    T_sl_sim = np.zeros((B, M))

    for b in range(B):
        # simulate new response under H0: y* ~ N(X β̂, σ̂² I)
        y_sim = X_with_intercept @ beta_hat + sigma_hat * np.random.randn(n)

        # refit OLS
        beta_sim = np.linalg.lstsq(X_with_intercept, y_sim, rcond=None)[0]
        y_hat_sim = X_with_intercept @ beta_sim
        residuals_sim = y_sim - y_hat_sim

        sigma2_sim = np.sum(residuals_sim ** 2) / (n - p - 1)
        sigma_sim = np.sqrt(sigma2_sim)

        try:
            XtX_inv_sim = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            H_sim = X_with_intercept @ XtX_inv_sim @ X_with_intercept.T
            h_ii_sim = np.diag(H_sim)
        except np.linalg.LinAlgError:
            h_ii_sim = np.full(n, (p + 1) / n)

        std_res_sim = residuals_sim / (sigma_sim * np.sqrt(1.0 - h_ii_sim + 1e-10))

        # T_RVF_sim
        T_rvf_sim[b, :] = loess_smooth(y_hat_sim, residuals_sim, grid_f, span=LOESS_SPAN)

        # Δ_QQ_sim
        d_ordered_sim = np.sort(std_res_sim)
        d_interp_sim = np.interp(q_theo_grid, q_theoretical_full, d_ordered_sim)
        T_qq_sim[b, :] = d_interp_sim - q_theo_grid

        # T_SL_sim
        sqrt_abs_std_sim = np.sqrt(np.abs(std_res_sim))
        T_sl_sim[b, :] = loess_smooth(y_hat_sim, sqrt_abs_std_sim, grid_f, span=LOESS_SPAN)

    return {
        "rvf_sim": T_rvf_sim,
        "qq_sim": T_qq_sim,
        "sl_sim": T_sl_sim,
        "q_theoretical": q_theo_grid,
    }


def compute_global_envelope(T_obs, T_sim, alpha=ALPHA):
    """
    Compute the global rank envelope and p-value using the
    maximum absolute deviation from the pointwise median.

    Parameters
    ----------
    T_obs : (m,) array — observed functional statistic
    T_sim : (B, m) array — simulated null statistics
    alpha : float — significance level

    Returns
    -------
    dict with keys: 'lower', 'upper', 'median', 'p_value', 'e_obs', 'e_crit'
    """
    B_sim = T_sim.shape[0]

    # pointwise median as central reference
    M_curve = np.median(T_sim, axis=0)

    # extremeness measures
    e_sim = np.max(np.abs(T_sim - M_curve), axis=1)
    e_obs = np.max(np.abs(T_obs - M_curve))

    # critical rank
    r_crit = int(np.ceil((1.0 - alpha) * (B_sim + 1)))
    # ascending sort
    e_sorted = np.sort(e_sim)
    # clamp r_crit to valid index range
    r_crit = min(r_crit, B_sim)
    e_crit = e_sorted[r_crit - 1] if r_crit > 0 else e_sorted[0]

    # envelope bounds
    lower = M_curve - e_crit
    upper = M_curve + e_crit

    # global p-value (descending rank)
    rank_obs = np.sum(e_sim >= e_obs) + 1
    p_val = rank_obs / (B_sim + 1)

    return {
        "lower": lower,
        "upper": upper,
        "median": M_curve,
        "p_value": p_val,
        "e_obs": e_obs,
        "e_crit": e_crit,
    }


# ============================================================
#  V. Visualization & Reporting
# ============================================================
def plot_diagnostic(ax, grid, T_obs, lower, upper, median_curve,
                    title, ylabel, highlight_rejections=True):
    """
    Draw a single diagnostic plot with:
      - grey shaded 95% global envelope
      - observed curve (solid line)
      - pointwise median (dashed line)
      - rejection points highlighted in red.
    """
    # envelope
    ax.fill_between(grid, lower, upper, color="grey", alpha=0.25,
                    label="95% Global Envelope")

    # median
    ax.plot(grid, median_curve, "b--", linewidth=1.5, label="Median (simulated)")

    # observed curve
    ax.plot(grid, T_obs, "k-", linewidth=2.0, label="Observed")

    # highlight points outside the envelope
    if highlight_rejections:
        outside = (T_obs < lower) | (T_obs > upper)
        if np.any(outside):
            ax.scatter(grid[outside], T_obs[outside],
                       color="red", s=20, zorder=5, label="Outside envelope")

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Fitted values" if "Residuals" in title else
                  "Theoretical quantiles" if "Q-Q" in title else
                  "Fitted values")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(labelsize=8)


def generate_report(fit_result, rvf_result, qq_result, sl_result,
                    X_test, y_test, obs_stats):
    """Generate the final integrated diagnostic report (report.md)."""
    # Test set evaluation
    n_test = X_test.shape[0]
    X_test_with_intercept = np.column_stack([np.ones(n_test), X_test])
    y_pred = X_test_with_intercept @ fit_result["beta_hat"]
    r2_test = r2_score(y_test, y_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

    # Build report
    lines = []
    lines.append("# Linear Regression Diagnostic Report")
    lines.append("")
    lines.append("## Global Rank Envelope Diagnostic Framework")
    lines.append("")
    lines.append(
        "This report applies the Global Rank Envelope procedure "
        "(Myllymäki et al., 2017) to formally test the assumptions "
        "of a linear regression model fitted on synthetic data."
    )
    lines.append("")

    # --- Model summary ---
    lines.append("## Model Summary")
    lines.append("")
    lines.append(f"- **Number of training samples:** {len(fit_result['y_hat'])}")
    lines.append(f"- **Number of test samples:** {len(y_test)}")
    lines.append(f"- **Number of features:** {len(fit_result['coef'])}")
    lines.append("")
    lines.append("### Coefficients")
    lines.append("")
    lines.append("| Term | Coefficient |")
    lines.append("|------|-------------|")
    lines.append(f"| Intercept | {fit_result['intercept']:.4f} |")
    for j, c in enumerate(fit_result["coef"]):
        lines.append(f"| x{j + 1} | {c:.4f} |")
    lines.append("")
    lines.append("### Goodness-of-Fit (Training Set)")
    lines.append("")
    lines.append(f"- **R² (train):** {fit_result['r2_train']:.4f}")
    lines.append(f"- **F-statistic:** {fit_result['f_statistic']:.4f}")
    lines.append(f"- **F p-value:** {fit_result['f_pval']:.6e}")
    lines.append(f"- **σ̂ (residual std error):** {fit_result['sigma_hat']:.4f}")
    lines.append("")

    # --- Diagnostic plots ---
    lines.append("## Diagnostic Plots")
    lines.append("")
    lines.append("Each plot shows the observed functional statistic (black solid line), "
                 "the pointwise median of 1000 simulated null curves (blue dashed line), "
                 "and the 95% global envelope (grey shaded band). Red points indicate "
                 "where the observed curve exits the envelope.")
    lines.append("")

    lines.append("### 1. Linearity — Residuals vs. Fitted (R v. F)")
    lines.append("")
    lines.append("![Residuals vs. Fitted](images/residuals_vs_fitted.png)")
    lines.append("")
    lines.append(f"- **p_RVF = {rvf_result['p_value']:.4f}**")
    conclusion_rvf = "PASS" if rvf_result["p_value"] >= ALPHA else "REJECT"
    lines.append(f"- **Conclusion:** {conclusion_rvf} — linearity assumption is "
                 f"{'not rejected' if conclusion_rvf == 'PASS' else 'rejected'} (α = {ALPHA}).")
    lines.append("")

    lines.append("### 2. Normality — Q-Q Plot (Δ_QQ)")
    lines.append("")
    lines.append("![Q-Q Plot](images/qq_plot.png)")
    lines.append("")
    lines.append(f"- **p_QQ = {qq_result['p_value']:.4f}**")
    conclusion_qq = "PASS" if qq_result["p_value"] >= ALPHA else "REJECT"
    lines.append(f"- **Conclusion:** {conclusion_qq} — normality assumption is "
                 f"{'not rejected' if conclusion_qq == 'PASS' else 'rejected'} (α = {ALPHA}).")
    lines.append("")

    lines.append("### 3. Homoscedasticity — Scale-Location (S v. L)")
    lines.append("")
    lines.append("![Scale-Location](images/scale_location.png)")
    lines.append("")
    lines.append(f"- **p_SL = {sl_result['p_value']:.4f}**")
    conclusion_sl = "PASS" if sl_result["p_value"] >= ALPHA else "REJECT"
    lines.append(f"- **Conclusion:** {conclusion_sl} — homoscedasticity assumption is "
                 f"{'not rejected' if conclusion_sl == 'PASS' else 'rejected'} (α = {ALPHA}).")
    lines.append("")

    # --- Summary table ---
    lines.append("## Hypothesis Test Summary")
    lines.append("")
    lines.append("| Assumption | Test | p-value | α | Verdict |")
    lines.append("|------------|------|---------|---|---------|")
    lines.append(f"| Linearity | Residuals vs. Fitted | {rvf_result['p_value']:.4f} | {ALPHA} | {conclusion_rvf} |")
    lines.append(f"| Normality | Q-Q Plot (Δ_QQ) | {qq_result['p_value']:.4f} | {ALPHA} | {conclusion_qq} |")
    lines.append(f"| Homoscedasticity | Scale-Location | {sl_result['p_value']:.4f} | {ALPHA} | {conclusion_sl} |")
    lines.append("")

    # --- Test set evaluation ---
    lines.append("## Test Set Evaluation")
    lines.append("")
    lines.append(f"- **R² (test):** {r2_test:.4f}")
    lines.append(f"- **RMSE (test):** {rmse_test:.4f}")
    lines.append("")

    # --- Overall conclusion ---
    lines.append("## Overall Conclusion")
    lines.append("")

    all_assumptions_pass = (
        rvf_result["p_value"] >= ALPHA
        and qq_result["p_value"] >= ALPHA
        and sl_result["p_value"] >= ALPHA
    )
    f_test_significant = fit_result["f_pval"] < ALPHA
    r2_adequate = fit_result["r2_train"] > 0.7

    if all_assumptions_pass and f_test_significant and r2_adequate:
        lines.append("**The linear model is considered adequate.**")
        lines.append("")
        lines.append("- All three regression assumptions are not rejected at α = 0.05.")
        lines.append("- The overall F-test is significant.")
        lines.append(f"- Training R² = {fit_result['r2_train']:.4f} > 0.7.")
        lines.append(f"- Test R² = {r2_test:.4f} and RMSE = {rmse_test:.4f} "
                     "confirm predictive consistency.")
    else:
        lines.append("**The linear model requires further investigation.**")
        lines.append("")
        if not all_assumptions_pass:
            lines.append("- One or more regression assumptions are violated at α = 0.05.")
        if not f_test_significant:
            lines.append("- The overall F-test is not significant.")
        if not r2_adequate:
            lines.append(f"- Training R² = {fit_result['r2_train']:.4f} ≤ 0.7.")
    lines.append("")

    # --- Technical parameters ---
    lines.append("## Technical Parameters")
    lines.append("")
    lines.append(f"- **Number of simulations (B):** {B}")
    lines.append(f"- **Evaluation grid points (m):** {M}")
    lines.append(f"- **LOESS span:** {LOESS_SPAN}")
    lines.append(f"- **LOESS polynomial degree:** {LOESS_DEGREE}")
    lines.append(f"- **Significance level (α):** {ALPHA}")
    lines.append(f"- **Train/Test split:** {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}")
    lines.append(f"- **Random seed:** {RANDOM_STATE}")
    lines.append("")
    lines.append("---")
    lines.append("*Report generated by Global Rank Envelope Diagnostic Framework.*")
    lines.append("")

    report_text = "\n".join(lines)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def save_outputs(fit_result, rvf_result, qq_result, sl_result, obs_stats):
    """Save numeric results to CSV files in outputs/."""
    # Model coefficients
    coef_df = pd.DataFrame({
        "term": ["Intercept"] + [f"x{j+1}" for j in range(len(fit_result["coef"]))],
        "coefficient": [fit_result["intercept"]] + list(fit_result["coef"]),
    })
    coef_df.to_csv(os.path.join(OUTPUTS_DIR, "coefficients.csv"), index=False)

    # P-values
    pval_df = pd.DataFrame({
        "test": ["Residuals vs. Fitted", "Q-Q Plot", "Scale-Location"],
        "p_value": [rvf_result["p_value"], qq_result["p_value"], sl_result["p_value"]],
    })
    pval_df.to_csv(os.path.join(OUTPUTS_DIR, "p_values.csv"), index=False)

    # Envelope curves
    for name, res, grid in [
        ("rvf", rvf_result, obs_stats["grid_f"]),
        ("qq", qq_result, obs_stats["q_theoretical"]),
        ("sl", sl_result, obs_stats["grid_f"]),
    ]:
        env_df = pd.DataFrame({
            "grid": grid,
            "observed": obs_stats[name.replace("rvf", "rvf").replace("qq", "qq_delta").replace("sl", "sl")],
            "lower": res["lower"],
            "upper": res["upper"],
            "median": res["median"],
        })
        env_df.to_csv(os.path.join(OUTPUTS_DIR, f"envelope_{name}.csv"), index=False)

    print(f"[save] coefficients -> outputs/coefficients.csv")
    print(f"[save] p-values    -> outputs/p_values.csv")
    print(f"[save] envelopes   -> outputs/envelope_{{rvf,qq,sl}}.csv")


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 60)
    print("Global Rank Envelope Diagnostic Framework")
    print("=" * 60)

    # ---------- I. Data preparation ----------
    print("\n[I] Loading and splitting data ...")
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data()
    print(f"    Train: {X_train.shape[0]} samples")
    print(f"    Test:  {X_test.shape[0]} samples")

    # ---------- II. Model fitting ----------
    print("[II] Fitting OLS model ...")
    fit_result = fit_ols(X_train, y_train)
    print(f"    R² (train) = {fit_result['r2_train']:.4f}")
    print(f"    F-statistic = {fit_result['f_statistic']:.4f} (p = {fit_result['f_pval']:.6e})")
    print(f"    σ̂ = {fit_result['sigma_hat']:.4f}")

    # Common evaluation grid
    y_hat = fit_result["y_hat"]
    grid_f = np.linspace(np.min(y_hat), np.max(y_hat), M)

    # Observed statistics
    print("    Computing observed diagnostic curves ...")
    obs_stats = compute_observed_statistics(fit_result, grid_f)

    # ---------- III. Simulation & envelope construction ----------
    print(f"[III] Simulating {B} null datasets (this may take a while) ...")
    sim_results = simulate_null_statistics(X_train, fit_result, grid_f)

    print("    Computing global rank envelopes ...")
    rvf_result = compute_global_envelope(obs_stats["rvf"], sim_results["rvf_sim"])
    qq_result = compute_global_envelope(obs_stats["qq_delta"], sim_results["qq_sim"])
    sl_result = compute_global_envelope(obs_stats["sl"], sim_results["sl_sim"])

    # For Q-Q plot, convert envelope from difference space to original quantile space
    q_theo = obs_stats["q_theoretical"]
    qq_lower_orig = q_theo + qq_result["lower"]
    qq_upper_orig = q_theo + qq_result["upper"]
    qq_median_orig = q_theo + qq_result["median"]

    print(f"\n    p_RVF = {rvf_result['p_value']:.4f}")
    print(f"    p_QQ  = {qq_result['p_value']:.4f}")
    print(f"    p_SL  = {sl_result['p_value']:.4f}")

    # ---------- IV. Visualization ----------
    print("[IV] Generating diagnostic plots ...")

    # 1) Residuals vs Fitted
    fig_rvf, ax_rvf = plt.subplots(figsize=(8, 5))
    plot_diagnostic(
        ax_rvf, grid_f, obs_stats["rvf"],
        rvf_result["lower"], rvf_result["upper"],
        rvf_result["median"],
        title="Linearity Test: Residuals vs. Fitted",
        ylabel="Residuals",
    )
    fig_rvf.tight_layout()
    fig_rvf.savefig(os.path.join(IMAGES_DIR, "residuals_vs_fitted.png"), dpi=150)
    plt.close(fig_rvf)

    # 2) Q-Q Plot (on original quantile scale)
    fig_qq, ax_qq = plt.subplots(figsize=(8, 5))
    ax_qq.fill_between(q_theo, qq_lower_orig, qq_upper_orig,
                        color="grey", alpha=0.25, label="95% Global Envelope")
    ax_qq.plot(q_theo, qq_median_orig, "b--", linewidth=1.5,
               label="Median (simulated)")
    ax_qq.plot(q_theo, obs_stats["d_interp"], "k-", linewidth=2.0,
               label="Observed")
    # Rejection points
    outside_qq = (obs_stats["d_interp"] < qq_lower_orig) | (obs_stats["d_interp"] > qq_upper_orig)
    if np.any(outside_qq):
        ax_qq.scatter(q_theo[outside_qq], obs_stats["d_interp"][outside_qq],
                      color="red", s=20, zorder=5, label="Outside envelope")
    ax_qq.plot(q_theo, q_theo, "gray", linestyle=":", linewidth=0.8,
               label="Identity line")
    ax_qq.set_xlabel("Theoretical quantiles")
    ax_qq.set_ylabel("Ordered standardized residuals")
    ax_qq.set_title("Normality Test: Q-Q Plot")
    ax_qq.legend(fontsize=8, loc="lower right")
    ax_qq.tick_params(labelsize=8)
    fig_qq.tight_layout()
    fig_qq.savefig(os.path.join(IMAGES_DIR, "qq_plot.png"), dpi=150)
    plt.close(fig_qq)

    # 3) Scale-Location
    fig_sl, ax_sl = plt.subplots(figsize=(8, 5))
    plot_diagnostic(
        ax_sl, grid_f, obs_stats["sl"],
        sl_result["lower"], sl_result["upper"],
        sl_result["median"],
        title="Homoscedasticity Test: Scale-Location",
        ylabel="√|Standardized residuals|",
    )
    fig_sl.tight_layout()
    fig_sl.savefig(os.path.join(IMAGES_DIR, "scale_location.png"), dpi=150)
    plt.close(fig_sl)

    print(f"    Figures saved to report/images/")

    # ---------- V. Reporting ----------
    print("[V] Generating diagnostic report ...")
    save_outputs(fit_result, rvf_result, qq_result, sl_result, obs_stats)

    report = generate_report(fit_result, rvf_result, qq_result, sl_result,
                             X_test, y_test, obs_stats)
    print("    Report saved to report/report.md")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  R² (train): {fit_result['r2_train']:.4f}")
    print(f"  F-test:     F={fit_result['f_statistic']:.2f}, p={fit_result['f_pval']:.6e}")
    print(f"  p_RVF:      {rvf_result['p_value']:.4f}")
    print(f"  p_QQ:       {qq_result['p_value']:.4f}")
    print(f"  p_SL:       {sl_result['p_value']:.4f}")
    test_r2 = r2_score(y_test, np.column_stack([np.ones(len(y_test)), X_test]) @ fit_result["beta_hat"])
    test_rmse = np.sqrt(mean_squared_error(y_test, np.column_stack([np.ones(len(y_test)), X_test]) @ fit_result["beta_hat"]))
    print(f"  R² (test):  {test_r2:.4f}")
    print(f"  RMSE (test):{test_rmse:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
