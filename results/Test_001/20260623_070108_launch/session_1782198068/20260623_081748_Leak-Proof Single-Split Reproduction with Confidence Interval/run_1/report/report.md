# Leak-Proof Single-Split Reproduction with Confidence Interval

## Linear Regression on Synthetic Data — Reproduction Report

### 1 Overview

This report presents a reproduction of a linear regression analysis on a synthetic
dataset containing **n = 1000** samples with **d = 5** features.  The methodology
follows a **leak-proof evaluation protocol**: a single 80/20 train-test split with
feature standardization parameters estimated **exclusively from the training set**,
an ordinary least squares (OLS) model fitted on the standardized training data,
and a 95% confidence interval for the true test MSE constructed via the
t-distribution.

| Setting | Value |
|---|---|
| Random seed | 42 |
| Training samples | 800 (80%) |
| Test samples | 200 (20%) |
| Model | OLS Linear Regression (fit_intercept=True) |
| Preprocessing | Z-score standardization (fit on train only) |
| Inference method | 95% t-confidence interval on squared errors |

---

### 2 Data Preprocessing

The 5 features (`x1` – `x5`) were standardized using the training-set mean and
sample standard deviation:

| Feature | Train Mean | Train Std |
|---------|-----------|-----------|
| x1 | 0.4949 | 0.2884 |
| x2 | 0.4899 | 0.2860 |
| x3 | 0.4960 | 0.2939 |
| x4 | 0.5167 | 0.2774 |
| x5 | 0.5061 | 0.2874 |

A small constant ε = 10⁻⁸ was added to the standard deviation to prevent division
by zero.  Standardization parameters were saved to
`outputs/standardization_params.csv`.

---

### 3 Model

Fitted OLS with an intercept term on the standardized training set
(800 samples, 5 features).  Coefficients:

| Term | Coefficient |
|------|-----------|
| Intercept | 3.7211 |
| x1 | 0.5812 |
| x2 | 0.8640 |
| x3 | 0.2948 |
| x4 | 0.1392 |
| x5 | 0.2859 |

The coefficient magnitudes indicate that features **x1** and **x2** have the
strongest influence on the target variable.  The intercept reflects the mean of
`y` (since features are standardized to zero mean).

![Model Coefficients](images/model_coefficients.png)

*Figure 1: Bar plot of the trained OLS model coefficients (intercept + 5 features).*

---

### 4 Test Performance

| Metric | Value |
|--------|-------|
| **MSE**  | **0.010758** |
| RMSE  | 0.103722 |
| R²    | **0.9920** |
| MAE   | 0.082197 |

The model achieves an **R² of 0.992**, indicating excellent predictive
performance — the linear model explains over 99% of the variance in the target
variable on the held-out test set.  The small MSE (0.0108) confirms that
prediction errors are tightly concentrated near zero.

---

### 5 Confidence Interval for True MSE

A 95% confidence interval for the expected generalization MSE was constructed
from the **200 test squared errors** using the Student's t-distribution:

- Sample mean (MSE):  **0.010758**
- Sample std (sₑ):     **0.015412**
- t-critical (df=199):  **1.9720**
- CI half-width:        **0.002149**
- **95% CI**:  **[0.008609, 0.012907]**

This interval accounts for sampling variability in the finite test set.  The
relatively narrow width (≈±20% of the point estimate) confirms that the MSE
estimate is reasonably precise with 200 test samples.

![Squared Errors with Confidence Interval](images/squared_errors_ci.png)

*Figure 2: Distribution of squared prediction errors on the test set.  The
vertical lines mark the sample mean MSE (dark red) and the 95% confidence
interval bounds (orange dashed).*

---

### 6 Reproduction Criterion

The paper's reported MSE value was not available in the provided materials.
When it becomes available, reproduction is deemed **successful** if it falls
within the computed 95% CI **[0.008609, 0.012907]**.

In the absence of a paper-reported value, the experiment establishes a rigorous
statistical benchmark.  Any future claim of a specific MSE for this dataset can
be evaluated against this interval.

---

### 7 Diagnostic Visualizations

#### 7.1 Predicted vs Actual

![Predicted vs Actual](images/predicted_vs_actual.png)

*Figure 3: Scatter plot of predicted versus actual target values on the test
set.  Points lie close to the diagonal (red dashed line), confirming accurate
predictions.*

#### 7.2 Residual Analysis

![Residual Analysis](images/residual_analysis.png)

*Figure 4: Left — histogram of residuals with a fitted normal distribution.
Right — Q-Q plot against the normal distribution.  Residuals appear
approximately normally distributed with near-zero mean, supporting the validity
of the OLS assumptions.*

#### 7.3 Predictions vs Actuals (Sequential)

![Predictions vs Actuals Scatter](images/predictions_vs_actuals_scatter.png)

*Figure 5: Sequential comparison of predicted (red crosses) and actual (blue
circles) values for each test sample.  Gray vertical lines highlight individual
prediction errors.*

---

### 8 Discussion

**Leak-proof design.**  By computing standardization parameters **only on the
training set**, the test set remains completely unseen during preprocessing,
ensuring an unbiased estimate of generalization performance.  This avoids the
common pitfall of data leakage where test data influences the model through
global scaling statistics.

**Statistical validity.**  The t-distribution-based confidence interval accounts
for the finite test set size (m = 200) and provides a principled decision rule
for reproduction: if a paper's reported MSE falls within the interval, the
reproduction is considered successful.  This is superior to comparing point
estimates or relying on repeated cross-validation with its flawed independence
assumptions (Nadeau & Bengio, 2003).

**Model performance.**  With an R² of 0.992 and MSE of 0.0108, the linear model
fits the synthetic data nearly perfectly.  The coefficients reveal that `x2` and
`x1` are the most predictive features, while `x4` contributes the least.

---

### 9 Files Produced

| File | Description |
|------|-------------|
| `code/experiment.py` | Complete reproduction script |
| `outputs/results.json` | All numerical results (metrics, CI, coefficients) |
| `outputs/standardization_params.csv` | Training-set means and stds per feature |
| `outputs/model_coefficients.csv` | Trained model coefficients |
| `report/images/predicted_vs_actual.png` | Predicted vs actual scatter |
| `report/images/residual_analysis.png` | Residual histogram + Q-Q plot |
| `report/images/squared_errors_ci.png` | Squared errors with 95% CI |
| `report/images/model_coefficients.png` | Coefficient bar plot |
| `report/images/predictions_vs_actuals_scatter.png` | Sequential prediction comparison |
| `report/report.md` | This report |
