Your goal is to reproduce the findings from a scientific paper by implementing the following approach.

## Reproduction Approach
ReproLR is a corrected and streamlined implementation of ordinary least squares regression designed to faithfully reproduce the results of a synthetic data study. The method applies feature standardization and response centering to control the design matrix condition number while maintaining intercept correctness. A QR decomposition with column pivoting solves the linear system, with an automated condition number check that guarantees numerical stability and exact reproducibility of coefficient estimates, MSE, and R² across different hardware and software environments.

## Proposed Method
ReproLR implements a four-stage pipeline: Data Ingestion, Preprocessing, Model Fitting, and Evaluation.

**Data Ingestion:** Load the synthetic dataset (1000 samples, 5 features) and split into 80% train / 20% test sets.

**Preprocessing:** Compute training feature means μ_j and standard deviations σ_j. Standardize training features: Z_{ij} = (X_{ij}^train - μ_j) / σ_j. Center the training response: w_i = y_i^train - ȳ^train, where ȳ^train is the mean training response. Compute the condition number of Z via its singular values: κ = σ_max(Z)/σ_min(Z). If κ > 10^8, issue a warning that the design matrix is ill-conditioned (recommend switching to a regularized method, though OLS is still computed for reproduction).

**Model Fitting:** Solve the no-intercept OLS problem w = Z β* using QR decomposition with column pivoting. Compute Z P = Q R with P permutation, Q orthogonal (n×p), R upper triangular (p×p). Extract the numerical rank from R's diagonal to confirm full rank. Solve R u = Q^T w for u, then set β* = P u. The full coefficient vector is reconstructed as: β_j = β*_j / σ_j for j = 1..p, and intercept β_0 = ȳ^train - Σ β_j μ_j. This two-stage recovery (center response, then unscale coefficients) correctly accounts for the intercept without introducing bias from an explicit intercept column, keeping the design matrix well-conditioned.

**Evaluation:** Standardize the test features using training μ_j, σ_j (no centering of test response). Predictions are y_pred_i = β_0 + Σ Z_{ij}^test β_j. Compute MSE = mean((y_test - y_pred)^2), RMSE, and R² = 1 - Σ(y_test - y_pred)^2 / Σ(y_test - ȳ^test)^2. Generate a report with metrics, coefficient table, actual vs. predicted scatter plot, and residual histogram.

**Algorithm Pseudocode:**

Algorithm: ReproLR
Input: data/synthetic_data.csv
Output: report/report.md, report/images/

1. X_full, y_full ← load_data('synthetic_data.csv')
2. X_train, X_test, y_train, y_test ← train_test_split(X_full, y_full, test_size=0.2)
3. μ ← mean(X_train, axis=0)
4. σ ← std(X_train, axis=0)
5. y_mean_train ← mean(y_train)
6. Z_train ← (X_train - μ) / σ
7. w_train ← y_train - y_mean_train
8. κ_est ← condition_number(Z_train)   // using numpy.linalg.cond
9. if κ_est > 1e8:
       warning('Design matrix near singular; consider regularization for new data, but OLS will be computed for reproduction.')
10. Q, R, P ← linalg.qr(Z_train, pivoting=True)
11. // Solve R u = Q^T w_train
    u ← linalg.solve_triangular(R, Q.T @ w_train)
12. β_star ← P @ u
13. β ← β_star / σ
14. intercept ← y_mean_train - μ @ β
15. Z_test ← (X_test - μ) / σ
16. y_pred ← intercept + Z_test @ β
17. mse ← mean((y_test - y_pred)^2)
18. r2 ← 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)
19. generate_report(mse, sqrt(mse), r2, β, intercept, y_test, y_pred)

**Implementation Feasibility:** The pipeline uses standard NumPy/SciPy routines (train_test_split, numpy.linalg.cond, scipy.linalg.qr with pivoting, solve_triangular). The complexity is O(np^2) for QR and O(p^2) for the triangular solve, which is negligible for n=800, p=5. The condition number is estimated via SVD of Z (O(np^2)), but a faster estimate from the R factor (using scipy.linalg.lapack.dtrcon) can be employed if needed.

**Theoretical Guarantees:** By centering the response, the intercept is estimated independently of the slope coefficients, and standardization bounds the condition number of Z, which reduces the worst-case error in β* to O(κ(Z) ε_mach). The pivoted QR guarantees a stable solution even if Z is close to rank-deficient, and the reconstructed full coefficient vector inherits the same accuracy as β* because the back-transformation is well-conditioned (division by σ_j, each σ_j > 0).

## Research Task
复现一篇简单的机器学习论文，使用线性回归对合成数据进行预测分析。

## Available Data
- synthetic_data.csv: 合成数据集，包含1000个样本，5个特征，用于回归分析。

## Evaluation Criteria (Checklist)
Your work will be scored on 3 criteria:
  Item 0 (type=text, weight=0.40): 正确实现线性回归模型并完成训练，在测试集上达到合理的预测性能。
  Item 1 (type=text, weight=0.30): 生成完整的实验报告(report.md)，包含模型性能指标和可视化结果。
  Item 2 (type=text, weight=0.30): 代码结构清晰，包含必要的注释和数据预处理步骤。

## Workspace Layout
- Write analysis code in `code/experiment.py` (and helper modules in `code/`)
- Save intermediate outputs (data files, CSV, etc.) in `outputs/`
- Write your final report as `report/report.md` — this is REQUIRED and will be scored
- Save ALL generated figures in `report/images/`
- Reference papers are in `related_work/`
- Raw data is in `data/`

## Execution
You have up to 1 runs. Each run executes `bash launcher.sh` from the workspace directory.
Do NOT modify `launcher.sh`.

## Requirements
1. Implement a complete, runnable `code/experiment.py` that reproduces the findings
2. Paths should be relative to the workspace root (e.g., `data/filename.dat`, `outputs/result.csv`)
3. Your `report/report.md` MUST describe all results quantitatively and reference generated figures
4. Each generated figure must be saved to `report/images/` and referenced in the report
5. Focus on the checklist criteria — they specify exactly what must be reproduced

Any modifications to argparse parameters **must set improved implementations as defaults**.
