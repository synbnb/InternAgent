"""
Leak-Proof Single-Split Reproduction with Confidence Interval
==============================================================
Reproduce the predictive performance of a linear regression model on
synthetic data.  Uses a single 80/20 train-test split with leak-proof
standardization and a 95% t-confidence interval for the test MSE.

Method reference
----------------
- Bishop (2006): Pattern Recognition and Machine Learning.
- Nadeau & Bengio (2003): Inference for the generalization error.

Implementation checklist
------------------------
1. Data loading & inspection (n=1000, d=5)
2. Deterministic train-test split (seed=42, 80/20)
3. Leak-proof Z-score standardization (fit on train only)
4. OLS linear regression training
5. Test-set evaluation (MSE, RMSE, R², MAE)
6. 95% t-confidence interval for true MSE
7. Residual diagnostics (normality test)
8. Visualization (5 figures)
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # non-interactive backend for headless env
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths (relative to workspace root, as required by instructions)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "synthetic_data.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
REPORT_DIR = os.path.join(ROOT, "report")
IMAGES_DIR = os.path.join(REPORT_DIR, "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_FRAC = 0.8
EPS = 1e-8                              # prevent division by zero
ALPHA = 0.05                            # significance level for 95% CI
CONFIDENCE_LEVEL = 1 - ALPHA            # 0.95


def main():
    # -----------------------------------------------------------------------
    # 1. Load and inspect data
    # -----------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    features = ["x1", "x2", "x3", "x4", "x5"]
    target = "y"

    X = df[features].values.astype(np.float64)   # (n, d)
    y = df[target].values.astype(np.float64)     # (n,)
    n = len(X)
    d = X.shape[1]

    print(f"Data loaded: {n} samples, {d} features")
    print(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Target range:  [{y.min():.4f}, {y.max():.4f}]")
    print(f"Target mean ± std: {y.mean():.4f} ± {y.std(ddof=1):.4f}")

    # -----------------------------------------------------------------------
    # 2. Leak-proof split (deterministic via fixed seed)
    # -----------------------------------------------------------------------
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.permutation(n)
    n_train = int(TRAIN_FRAC * n)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nTrain set: {len(train_idx)} samples")
    print(f"Test set:  {len(test_idx)} samples")

    # Check target distribution consistency across splits
    print(f"  y_train: mean={y_train.mean():.4f}, std={y_train.std(ddof=1):.4f}")
    print(f"  y_test:  mean={y_test.mean():.4f}, std={y_test.std(ddof=1):.4f}")

    # -----------------------------------------------------------------------
    # 3. Leak-proof standardization (fit ONLY on training data)
    # -----------------------------------------------------------------------
    mu = X_train.mean(axis=0)                    # (d,) — training mean
    sigma = X_train.std(axis=0, ddof=1) + EPS    # (d,) — sample std + epsilon

    X_train_scaled = (X_train - mu) / sigma
    X_test_scaled = (X_test - mu) / sigma

    # Save standardization parameters
    std_params = pd.DataFrame({
        "feature": features,
        "mean": mu,
        "std": sigma
    })
    std_params.to_csv(os.path.join(OUTPUT_DIR, "standardization_params.csv"),
                      index=False)
    print("\nStandardization parameters (fitted on train set):")
    for i, feat in enumerate(features):
        print(f"  {feat}: μ={mu[i]:.4f}, σ={sigma[i]:.4f}")

    # -----------------------------------------------------------------------
    # 4. Train OLS linear regression model
    # -----------------------------------------------------------------------
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train_scaled, y_train)

    # Extract and save coefficients
    coef_names = ["intercept"] + features
    coef_values = [model.intercept_] + list(model.coef_)
    coef_df = pd.DataFrame({
        "feature": coef_names,
        "coefficient": coef_values
    })
    coef_df.to_csv(os.path.join(OUTPUT_DIR, "model_coefficients.csv"),
                   index=False)
    print("\nTrained model coefficients:")
    for name, val in zip(coef_names, coef_values):
        print(f"  {name}: {val:.6f}")

    # -----------------------------------------------------------------------
    # 5. Evaluate on test set (multiple metrics)
    # -----------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)

    squared_errors = (y_test - y_pred) ** 2
    mse_test = float(squared_errors.mean())
    rmse_test = float(np.sqrt(mse_test))
    r2_test = float(r2_score(y_test, y_pred))
    mae_test = float(mean_absolute_error(y_test, y_pred))

    # Additional diagnostics
    residuals = y_test - y_pred
    max_error = float(np.max(np.abs(residuals)))
    residual_std = float(residuals.std(ddof=1))

    print(f"\n{'='*60}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*60}")
    print(f"  MSE:           {mse_test:.6f}")
    print(f"  RMSE:          {rmse_test:.6f}")
    print(f"  R²:            {r2_test:.6f}")
    print(f"  MAE:           {mae_test:.6f}")
    print(f"  Max |error|:   {max_error:.6f}")
    print(f"  Residual std:  {residual_std:.6f}")

    # -----------------------------------------------------------------------
    # 6. 95% t-confidence interval for the expected test MSE
    # -----------------------------------------------------------------------
    m = len(y_test)
    s_e = float(squared_errors.std(ddof=1))           # sample std of squared errors
    t_crit = stats.t.ppf(1 - ALPHA / 2, df=m - 1)    # ≈ 1.972 for m = 200
    ci_half = t_crit * s_e / np.sqrt(m)

    ci_lower = mse_test - ci_half
    ci_upper = mse_test + ci_half

    print(f"\n{'='*60}")
    print("CONFIDENCE INTERVAL FOR TRUE MSE")
    print(f"{'='*60}")
    print(f"  Sample MSE (ē):        {mse_test:.6f}")
    print(f"  Std of squared errors: {s_e:.6f}")
    print(f"  t-critical (df={m-1}):  {t_crit:.4f}")
    print(f"  CI half-width:         {ci_half:.6f}")
    print(f"  95% CI:               [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  CI relative width:     ±{ci_half/mse_test*100:.1f}% of point estimate")

    # -----------------------------------------------------------------------
    # 7. Residual diagnostics — normality test
    # -----------------------------------------------------------------------
    # Shapiro-Wilk test for normality of residuals
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # SW limited to n<5000

    # D'Agostino-Pearson omnibus test
    dagostino_stat, dagostino_p = stats.normaltest(residuals)

    print(f"\n{'='*60}")
    print("RESIDUAL DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"  Residual mean:         {residuals.mean():.6f} (should be ~0)")
    print(f"  Residual std:          {residual_std:.6f}")
    print(f"  Residual skewness:     {stats.skew(residuals):.4f}")
    print(f"  Residual kurtosis:     {stats.kurtosis(residuals):.4f}")
    print(f"  Shapiro-Wilk:          W={shapiro_stat:.4f}, p={shapiro_p:.6f}")
    print(f"  D'Agostino-Pearson:    χ²={dagostino_stat:.4f}, p={dagostino_p:.6f}")

    normality_ok = shapiro_p > ALPHA
    print(f"  → Residuals {'appear normal (p>0.05)' if normality_ok else 'deviate from normality (p<=0.05)'}")

    # -----------------------------------------------------------------------
    # 8. Collect all results into a structured dict
    # -----------------------------------------------------------------------
    results = {
        "n_samples": int(n),
        "n_features": int(d),
        "n_train": int(n_train),
        "n_test": int(m),
        "random_seed": RANDOM_SEED,
        "train_fraction": TRAIN_FRAC,
        "model_type": "LinearRegression (OLS, fit_intercept=True)",
        "standardization": {
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "epsilon": EPS,
            "fitted_on": "train_only"
        },
        "coefficients": {
            "intercept": float(model.intercept_),
            "coef": model.coef_.tolist(),
            "feature_names": features
        },
        "metrics": {
            "mse": mse_test,
            "rmse": rmse_test,
            "r2": r2_test,
            "mae": mae_test,
            "max_error": max_error,
            "residual_std": residual_std
        },
        "confidence_interval_95": {
            "lower": ci_lower,
            "upper": ci_upper,
            "half_width": ci_half,
            "t_critical": float(t_crit),
            "degrees_of_freedom": m - 1,
            "std_squared_errors": s_e,
            "method": "t-distribution on test squared errors"
        },
        "residual_diagnostics": {
            "mean": float(residuals.mean()),
            "std": residual_std,
            "skewness": float(stats.skew(residuals)),
            "kurtosis": float(stats.kurtosis(residuals)),
            "shapiro_wilk": {
                "statistic": float(shapiro_stat),
                "p_value": float(shapiro_p)
            },
            "dagostino_pearson": {
                "statistic": float(dagostino_stat),
                "p_value": float(dagostino_p)
            },
            "is_normal": bool(normality_ok)
        }
    }

    # -----------------------------------------------------------------------
    # 8a. Reproduction criterion (paper MSE not available)
    # -----------------------------------------------------------------------
    PAPER_MSE = None   # set to float to enable reproduction check
    if PAPER_MSE is not None:
        results["paper_mse"] = PAPER_MSE
        results["reproduction_success"] = (ci_lower <= PAPER_MSE <= ci_upper)
        print(f"\nReproduction check: paper MSE={PAPER_MSE}")
        print(f"  → {'SUCCESS' if results['reproduction_success'] else 'FAIL'}")
    else:
        print("\nReproduction check: paper MSE not available (skipped)")

    # Save results
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to outputs/results.json")

    # -----------------------------------------------------------------------
    # 9. Visualizations (5+ figures saved to report/images/)
    # -----------------------------------------------------------------------

    # 9a. Predicted vs Actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Ideal")
    # Add R² annotation
    ax.text(0.05, 0.95, f"R² = {r2_test:.4f}\nMSE = {mse_test:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlabel("Actual y")
    ax.set_ylabel("Predicted y")
    ax.set_title("Predicted vs Actual (Test Set)")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "predicted_vs_actual.png"), dpi=150)
    plt.close(fig)
    print("Figure saved: predicted_vs_actual.png")

    # 9b. Residuals distribution + Q-Q plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram with normal overlay
    axes[0].hist(residuals, bins=20, density=True, alpha=0.7, color="steelblue",
                 edgecolor="white")
    x_grid = np.linspace(residuals.min(), residuals.max(), 200)
    res_mean = residuals.mean()
    res_std = residuals.std(ddof=1)
    axes[0].plot(x_grid, stats.norm.pdf(x_grid, res_mean, res_std),
                 "r-", lw=2, label=f"N({res_mean:.3f}, {res_std:.3f})")
    axes[0].set_xlabel("Residual (y − ŷ)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Residual Distribution\nShapiro-Wilk p={shapiro_p:.4f}")
    axes[0].legend()

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Residuals)")
    axes[1].get_lines()[0].set_markerfacecolor("steelblue")
    axes[1].get_lines()[0].set_markeredgecolor("steelblue")
    axes[1].get_lines()[0].set_alpha(0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "residual_analysis.png"), dpi=150)
    plt.close(fig)
    print("Figure saved: residual_analysis.png")

    # 9c. Squared errors with confidence interval
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(squared_errors, bins=20, density=True, alpha=0.7, color="forestgreen",
            edgecolor="white")

    # Get y-limit for shading
    ylim = ax.get_ylim()

    # Annotate MSE and CI
    ax.axvline(mse_test, color="darkred", linestyle="-", lw=2.5,
               label=f"MSE = {mse_test:.4f}")
    ax.axvline(ci_lower, color="darkorange", linestyle="--", lw=2,
               label=f"CI lower = {ci_lower:.4f}")
    ax.axvline(ci_upper, color="darkorange", linestyle="--", lw=2,
               label=f"CI upper = {ci_upper:.4f}")
    ax.fill_betweenx([0, ylim[1]], ci_lower, ci_upper,
                     color="orange", alpha=0.12, label=f"95% CI region")

    # Reset ylim after fill
    ax.set_ylim(0, ylim[1])

    ax.set_xlabel("Squared Error")
    ax.set_ylabel("Density")
    ax.set_title(f"Squared Prediction Errors\n"
                 f"MSE = {mse_test:.4f}  "
                 f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "squared_errors_ci.png"), dpi=150)
    plt.close(fig)
    print("Figure saved: squared_errors_ci.png")

    # 9d. Feature coefficients bar plot
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_bar = ["gray"] + ["steelblue"] * d
    bars = ax.bar(coef_names, coef_values, color=colors_bar, edgecolor="white")
    # Annotate values on bars
    for bar, val in zip(bars, coef_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.02 if val >= 0 else -0.08),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Trained OLS Model Coefficients")
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "model_coefficients.png"), dpi=150)
    plt.close(fig)
    print("Figure saved: model_coefficients.png")

    # 9e. Predictions vs actuals (sequential)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(y_test)), y_test, "o", markersize=4, alpha=0.6,
            label="Actual", color="steelblue")
    ax.plot(range(len(y_pred)), y_pred, "x", markersize=4, alpha=0.6,
            label="Predicted", color="crimson")
    ax.vlines(range(len(y_test)), y_test, y_pred, alpha=0.15, color="gray")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("y")
    ax.set_title(f"Predictions vs Actuals (Test Set, MSE={mse_test:.4f})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "predictions_vs_actuals_scatter.png"),
                dpi=150)
    plt.close(fig)
    print("Figure saved: predictions_vs_actuals_scatter.png")

    # -----------------------------------------------------------------------
    # 10. Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(f"  Samples:              {n} total → {n_train} train + {m} test")
    print(f"  Features:             {d}")
    print(f"  Model:                OLS Linear Regression")
    print(f"  Test MSE:             {mse_test:.6f}")
    print(f"  Test R²:              {r2_test:.6f}")
    print(f"  95% CI for MSE:      [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  Residual normality:   {'YES' if normality_ok else 'NO'} (p={shapiro_p:.4f})")
    print(f"  Figures saved:        {len(os.listdir(IMAGES_DIR))} files in {IMAGES_DIR}")
    print(f"  Output files:          {OUTPUT_DIR}/")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    results = main()
