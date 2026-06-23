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
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ---------- paths (relative to workspace root) ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "synthetic_data.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
REPORT_DIR = os.path.join(ROOT, "report")
IMAGES_DIR = os.path.join(REPORT_DIR, "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------- configuration ----------
RANDOM_SEED = 42
TRAIN_FRAC = 0.8
EPS = 1e-8                              # prevent division by zero
ALPHA = 0.05                            # significance level for 95% CI

# ---------- 1. load data ----------
df = pd.read_csv(DATA_PATH)
features = ["x1", "x2", "x3", "x4", "x5"]
target = "y"

X = df[features].values.astype(np.float64)   # (n, d)
y = df[target].values.astype(np.float64)     # (n,)
n = len(X)
d = X.shape[1]

print(f"Data loaded: {n} samples, {d} features")

# ---------- 2. leak-proof split ----------
rng = np.random.RandomState(RANDOM_SEED)
indices = rng.permutation(n)
n_train = int(TRAIN_FRAC * n)

train_idx = indices[:n_train]
test_idx  = indices[n_train:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train set: {len(train_idx)} samples")
print(f"Test set:  {len(test_idx)} samples")

# ---------- 3. leak-proof standardization ----------
mu    = X_train.mean(axis=0)                  # (d,)
sigma = X_train.std(axis=0, ddof=1) + EPS     # (d,)  sample std + epsilon

X_train_scaled = (X_train - mu) / sigma
X_test_scaled  = (X_test  - mu) / sigma

# Save standardization parameters for reproducibility
std_params = pd.DataFrame({
    "feature": features,
    "mean": mu,
    "std": sigma
})
std_params.to_csv(os.path.join(OUTPUT_DIR, "standardization_params.csv"),
                  index=False)
print("Standardization parameters saved.")

# ---------- 4. train OLS model ----------
model = LinearRegression(fit_intercept=True)
model.fit(X_train_scaled, y_train)

# Extract coefficients
coef_df = pd.DataFrame({
    "feature": ["intercept"] + features,
    "coefficient": [model.intercept_] + list(model.coef_)
})
coef_df.to_csv(os.path.join(OUTPUT_DIR, "model_coefficients.csv"),
               index=False)
print("Model coefficients saved.")

# ---------- 5. evaluate on test set ----------
y_pred = model.predict(X_test_scaled)

squared_errors = (y_test - y_pred) ** 2
mse_test = float(squared_errors.mean())
rmse_test = float(np.sqrt(mse_test))
r2_test = float(r2_score(y_test, y_pred))
mae_test = float(np.mean(np.abs(y_test - y_pred)))

print(f"Test MSE:  {mse_test:.6f}")
print(f"Test RMSE: {rmse_test:.6f}")
print(f"Test R²:   {r2_test:.6f}")
print(f"Test MAE:  {mae_test:.6f}")

# ---------- 6. 95% confidence interval for the true MSE ----------
m = len(y_test)
s_e = float(squared_errors.std(ddof=1))          # sample std of squared errors
t_crit = stats.t.ppf(1 - ALPHA / 2, df=m - 1)    # ≈ 1.972 for m=200
ci_half = t_crit * s_e / np.sqrt(m)

ci_lower = mse_test - ci_half
ci_upper = mse_test + ci_half

print(f"\n95% CI for true MSE: [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"t-critical ({m-1} df): {t_crit:.4f}")
print(f"Std of squared errors (s_e): {s_e:.6f}")

# ---------- 7. reproduction criterion (paper MSE not available) ----------
# NOTE: The paper MSE value is not provided in the available materials.
# The pseudocode in the instructions uses 0.0 as a placeholder.
# If/when the paper MSE becomes available, set PAPER_MSE below.
PAPER_MSE = None   # set to float to enable reproduction check

results = {
    "n_samples": int(n),
    "n_features": int(d),
    "n_train": int(n_train),
    "n_test": int(m),
    "random_seed": RANDOM_SEED,
    "model_type": "LinearRegression (OLS)",
    "standardization": {"mu": mu.tolist(), "sigma": sigma.tolist()},
    "coefficients": {"intercept": float(model.intercept_),
                     "coef": model.coef_.tolist()},
    "metrics": {
        "mse": mse_test,
        "rmse": rmse_test,
        "r2": r2_test,
        "mae": mae_test,
    },
    "confidence_interval_95": {
        "lower": ci_lower,
        "upper": ci_upper,
        "t_critical": float(t_crit),
        "std_squared_errors": s_e,
    },
}

if PAPER_MSE is not None:
    results["paper_mse"] = PAPER_MSE
    results["reproduction_success"] = (ci_lower <= PAPER_MSE <= ci_upper)

with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to outputs/results.json")

# ---------- 8. visualizations ----------

# 8a. Predicted vs Actual
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Ideal")
ax.set_xlabel("Actual y")
ax.set_ylabel("Predicted y")
ax.set_title("Predicted vs Actual (Test Set)")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
fig.savefig(os.path.join(IMAGES_DIR, "predicted_vs_actual.png"), dpi=150)
plt.close(fig)
print("Figure saved: predicted_vs_actual.png")

# 8b. Residuals distribution
residuals = y_test - y_pred
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(residuals, bins=20, density=True, alpha=0.7, color="steelblue",
             edgecolor="white")
# Overlay normal fit
x_grid = np.linspace(residuals.min(), residuals.max(), 200)
res_mean = residuals.mean()
res_std = residuals.std(ddof=1)
axes[0].plot(x_grid, stats.norm.pdf(x_grid, res_mean, res_std),
             "r-", lw=2, label=f"N({res_mean:.3f}, {res_std:.3f})")
axes[0].set_xlabel("Residual (y - ŷ)")
axes[0].set_ylabel("Density")
axes[0].set_title("Residual Distribution")
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

# 8c. Squared errors with confidence interval
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(squared_errors, bins=20, density=True, alpha=0.7, color="forestgreen",
        edgecolor="white")
# Annotate the sample mean and CI
ax.axvline(mse_test, color="darkred", linestyle="-", lw=2.5,
           label=f"MSE = {mse_test:.4f}")
ax.axvline(ci_lower, color="darkorange", linestyle="--", lw=2,
           label=f"CI 95% lower = {ci_lower:.4f}")
ax.axvline(ci_upper, color="darkorange", linestyle="--", lw=2,
           label=f"CI 95% upper = {ci_upper:.4f}")
ax.fill_betweenx([0, ax.get_ylim()[1]], ci_lower, ci_upper,
                 color="orange", alpha=0.12, label="95% CI region")
ax.set_xlabel("Squared Error")
ax.set_ylabel("Density")
ax.set_title(f"Squared Prediction Errors\nMSE = {mse_test:.4f}  "
             f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(IMAGES_DIR, "squared_errors_ci.png"), dpi=150)
plt.close(fig)
print("Figure saved: squared_errors_ci.png")

# 8d. Feature coefficients bar plot
fig, ax = plt.subplots(figsize=(8, 4))
coef_names = ["Intercept"] + features
coef_vals = [model.intercept_] + list(model.coef_)
colors = ["gray"] + ["steelblue"] * d
ax.bar(coef_names, coef_vals, color=colors, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Coefficient Value")
ax.set_title("Trained Model Coefficients")
plt.tight_layout()
fig.savefig(os.path.join(IMAGES_DIR, "model_coefficients.png"), dpi=150)
plt.close(fig)
print("Figure saved: model_coefficients.png")

# 8e. Prediction error scatter
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(y_test)), y_test, "o", markersize=4, alpha=0.6,
        label="Actual", color="steelblue")
ax.plot(range(len(y_pred)), y_pred, "x", markersize=4, alpha=0.6,
        label="Predicted", color="crimson")
ax.vlines(range(len(y_test)), y_test, y_pred, alpha=0.2, color="gray")
ax.set_xlabel("Test Sample Index")
ax.set_ylabel("y")
ax.set_title("Predictions vs Actuals on Test Set")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(IMAGES_DIR, "predictions_vs_actuals_scatter.png"),
            dpi=150)
plt.close(fig)
print("Figure saved: predictions_vs_actuals_scatter.png")

# ---------- summary ----------
print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
print(f"  Test MSE:        {mse_test:.6f}")
print(f"  Test RMSE:       {rmse_test:.6f}")
print(f"  Test R²:         {r2_test:.6f}")
print(f"  Test MAE:        {mae_test:.6f}")
print(f"  95% CI for MSE:  [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"  CI half-width:   {ci_half:.6f}")
print(f"  Figures saved:   {IMAGES_DIR}")
print("=" * 60)
