Your goal is to reproduce the findings from a scientific paper by implementing the following approach.

## Reproduction Approach
This method offers a rigorous yet simple framework for reproducing the predictive performance of a linear regression model. It uses a single random split of the dataset into training (80%) and test (20%) sets. Feature standardization parameters (mean, standard deviation) are computed exclusively on the training set to prevent data leakage, and an ordinary least squares model is fitted. The test mean squared error (MSE) is computed, and a 95% confidence interval for the true prediction MSE is constructed from the test squared errors using a t-distribution. Reproduction is deemed successful if the paper's reported MSE falls within this interval, thereby accounting for sampling variability in a statistically valid manner without the pitfalls of repeated cross-validation.

## Proposed Method
### Method Details

1. **Data Split**  
   Partition the dataset \(\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}\) (\(n=1000\)) into a training set \(\mathcal{D}_{\text{train}}\) (80%, 800 samples) and a test set \(\mathcal{D}_{\text{test}}\) (20%, 200 samples) using a fixed random seed (e.g., 42) for reproducibility. The split is stratified if the target is continuous? Not needed; simple random split.

2. **Leak-Proof Preprocessing**  
   Compute the mean \(\mu_j\) and standard deviation \(\sigma_j\) for each feature \(j = 1,\ldots,d\) (\(d=5\)) using **only** the training samples:  
   \[
   \mu_j = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(\mathbf{x},y) \in \mathcal{D}_{\text{train}}} x_j, \quad 
   \sigma_j = \sqrt{\frac{1}{|\mathcal{D}_{\text{train}}|-1} \sum_{(\mathbf{x},y) \in \mathcal{D}_{\text{train}}} (x_j - \mu_j)^2 + \epsilon},
   \]
   where a small constant \(\epsilon\) (e.g., \(10^{-8}\)) prevents division by zero. Apply the transformation to all feature vectors in both training and test sets:  
   \[
   x_j' = \frac{x_j - \mu_j}{\sigma_j}.
   \]
   This ensures that no information from the test data influences the model's preprocessing step, preserving the integrity of the performance estimate.

3. **Model Training**  
   Fit an ordinary least squares (OLS) linear regression model on the standardized training set \(\mathcal{D}_{\text{train}}'\). The model \(f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}' + b\) minimizes the mean squared error:  
   \[
   \min_{\mathbf{w}, b} \frac{1}{N_{\text{train}}} \sum_{(\mathbf{x}',y) \in \mathcal{D}_{\text{train}}'} (y - (\mathbf{w}^\top \mathbf{x}' + b))^2.
   \]
   Solve via the normal equations: \(\mathbf{X}^\top \mathbf{X} \mathbf{w} = \mathbf{X}^\top \mathbf{y}\) after augmenting with a column of ones for the intercept. Because \(d=5\) and \(N_{\text{train}}=800\), this is computationally trivial.

4. **Test Performance Evaluation**  
   For each test sample \((\mathbf{x}_i', y_i) \in \mathcal{D}_{\text{test}}'\), compute the prediction \(\hat{y}_i = f(\mathbf{x}_i')\). The test MSE is:  
   \[
   \text{MSE}_{\text{test}} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2, \quad m = 200.
   \]

5. **Confidence Interval Construction**  
   Let \(e_i = (y_i - \hat{y}_i)^2\) be the squared prediction error for each test point. Treating \(\{e_i\}\) as a sample of i.i.d. draws from the distribution of squared errors, compute the sample mean \(\bar{e} = \text{MSE}_{\text{test}}\) and sample standard deviation \(s_e\). A \(95\%\) confidence interval for the true expected MSE is:  
   \[
   \left[\bar{e} - t_{0.975, m-1} \cdot \frac{s_e}{\sqrt{m}}, \; \bar{e} + t_{0.975, m-1} \cdot \frac{s_e}{\sqrt{m}}\right],
   \]
   where \(t_{0.975, m-1}\) is the \(97.5^{\text{th}}\) percentile of the Student's t-distribution with \(m-1\) degrees of freedom (approximately 1.972 for \(m=200\)). This interval accounts for the uncertainty in the MSE estimate due to the finite test set size.

6. **Reproduction Criterion**  
   Let \(\text{MSE}_{\text{paper}}\) be the value reported in the paper. If \(\text{MSE}_{\text{paper}}\) lies within the computed confidence interval, reproduction is successful; otherwise, it is not. The use of a confidence interval provides a clear statistical decision rule that accommodates sampling variability without assuming that repeated cross‑validation estimates are independent.

7. **Implementation Guidance**  
   - **Random Seed**: Set globally (e.g., `numpy.random.seed(42)`) before splitting to ensure exact reproducibility.
   - **Pseudocode**:
   ```
   import numpy as np
   from sklearn.linear_model import LinearRegression
   from scipy import stats

   # Load data (n=1000, d=5 features, target y)
   X, y = load_data('synthetic_data.csv')
   n = len(X)

   # Split indices with fixed seed
   rng = np.random.RandomState(42)
   indices = rng.permutation(n)
   n_train = int(0.8 * n)
   train_idx, test_idx = indices[:n_train], indices[n_train:]
   X_train, X_test = X[train_idx], X[test_idx]
   y_train, y_test = y[train_idx], y[test_idx]

   # Compute standardization parameters from training data only
   mu = X_train.mean(axis=0)
   sigma = X_train.std(axis=0, ddof=1) + 1e-8
   X_train_scaled = (X_train - mu) / sigma
   X_test_scaled = (X_test - mu) / sigma

   # Train OLS model
   model = LinearRegression(fit_intercept=True)
   model.fit(X_train_scaled, y_train)

   # Predict on test set
   y_pred = model.predict(X_test_scaled)
   squared_errors = (y_test - y_pred)**2
   mse_test = squared_errors.mean()

   # 95% confidence interval
   m = len(y_test)
   s_e = squared_errors.std(ddof=1)
   t_crit = stats.t.ppf(0.975, df=m-1)
   ci_lower = mse_test - t_crit * s_e / np.sqrt(m)
   ci_upper = mse_test + t_crit * s_e / np.sqrt(m)

   # Check reproduction
   paper_mse = 0.0  # Replace with actual value from paper
   success = ci_lower <= paper_mse <= ci_upper
   ```
   - **Computational Complexity**: O(nd^2 + d^3) for training, negligible.

8. **Key Theoretical Justifications**  
   - **Unbiased Preprocessing**: Because standardization parameters are fit on the training set only, the OLS model trained on \(\mathcal{D}_{\text{train}}'\) is independent of \(\mathcal{D}_{\text{test}}\). Consequently, \(\text{MSE}_{\text{test}}\) is an unbiased estimate of the true generalization error of the preprocessing–model pipeline (Bishop, 2006).
   - **Validity of the Confidence Interval**: The test squared errors \(e_i\) are not strictly independent (predictions share the same model), but for a fixed model trained on independent data, conditional on the training set, the test errors are exchangeable and the normal/t approximation provides an asymptotically valid interval (Lehmann & Casella, 1998). Simulation studies (Nadeau & Bengio, 2003) show that for stable learners like OLS, the naive interval has near‑nominal coverage when test set size is reasonable.
   - **Comparison to Repeated CV**: The method avoids the flawed independence assumption among repeat‑level MSE estimates (Critique 1), and the separate test set guarantees that no data leakage occurs (Critique 2). The confidence interval directly captures the uncertainty in the performance estimate, enabling a principled comparison without requiring knowledge of the sampling variability of the reported value.

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
You have up to 2 runs. Each run executes `bash launcher.sh` from the workspace directory.
Do NOT modify `launcher.sh`.

## Requirements
1. Implement a complete, runnable `code/experiment.py` that reproduces the findings
2. Paths should be relative to the workspace root (e.g., `data/filename.dat`, `outputs/result.csv`)
3. Your `report/report.md` MUST describe all results quantitatively and reference generated figures
4. Each generated figure must be saved to `report/images/` and referenced in the report
5. Focus on the checklist criteria — they specify exactly what must be reproduced

Any modifications to argparse parameters **must set improved implementations as defaults**.
