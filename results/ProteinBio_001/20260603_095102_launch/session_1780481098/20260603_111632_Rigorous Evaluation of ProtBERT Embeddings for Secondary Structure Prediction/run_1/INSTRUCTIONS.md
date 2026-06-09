Your goal is to reproduce the findings from a scientific paper by implementing the following approach.

## Reproduction Approach
This method provides a rigorous framework to reproduce the core finding that ProtBERT per-residue embeddings outperform traditional physicochemical features for secondary structure prediction. The framework includes proper embedding extraction, training of linear classifiers with nested cross-validation to avoid data leakage, a controlled fusion experiment testing the redundancy of explicit global context, and a diagnostic probing analysis using repeated K-fold cross-validation with exact formulas for global biophysical properties. All uncertainty is quantified via appropriate confidence intervals from cross-validation scores, making the results reliable even on a small pilot dataset of 100 proteins.

## Proposed Method
## Introduction
This method aims to systematically compare the quality of representations extracted from the protein language model ProtBERT against traditional hand-crafted features for the task of residue-level secondary structure prediction. To ensure robust conclusions from a limited dataset (100 proteins), we design a nested cross-validation pipeline with careful separation of training and test folds, both at the protein level and within each fold's preprocessing. The framework also incorporates a probing analysis to assess whether mean-pooled embeddings linearly encode global biophysical properties, and a fusion experiment to determine if such global information provides additional predictive power beyond the per-residue embeddings.

## Notation
- $N$: number of proteins (here $N=100$).
- Protein $i$ has sequence $s_i = (a_{i,1}, \dots, a_{i, L_i})$, where $a_{i,j}$ is an amino acid type and $L_i$ is sequence length.
- $\mathbf{E}_i = [\mathbf{e}_{i,1}, \dots, \mathbf{e}_{i, L_i}] \in \mathbb{R}^{L_i \times d}$: per-residue embeddings from ProtBERT (excluding special tokens), with $d=1024$.
- $\mathbf{z}_i = \frac{1}{L_i} \sum_{j=1}^{L_i} \mathbf{e}_{i,j} \in \mathbb{R}^d$: mean-pooled embedding for protein $i$.
- Hand-crafted feature vector for residue $j$ in protein $i$: $\mathbf{f}_{i,j} \in \mathbb{R}^{d_f}$, e.g., one-hot encoding of amino acid type (20 dim) concatenated with physicochemical descriptors (Kyte-Doolittle hydrophobicity, charge, etc., totaling $d_f=22$).
- Each residue has a secondary structure label $t_{i,j} \in \{H, E, C\}$ (helix, strand, coil).
- Global properties for protein $i$:
  - Amino acid composition $\mathbf{y}_i^{AA} \in \mathbb{R}^{20}$, vector of fractions for each standard amino acid.
  - Net charge at pH 7 $y_i^{\text{charge}} \in \mathbb{R}$.
  - Average hydrophobicity $y_i^{\text{hydro}} \in \mathbb{R}$.

## System Architecture
The pipeline consists of four stages:
1. **Embedding and Feature Extraction**: Frozen ProtBERT generates per-residue embeddings; hand-crafted features are computed per residue from amino acid scales.
2. **Secondary Structure Prediction Comparison**: Three models (embeddings only, hand-crafted only, concatenation of both) are trained and evaluated via nested CV to compare Q3 accuracy.
3. **Fusion Experiment** (optional): A linear classifier is trained on per-residue embeddings augmented with a global context vector (mean-pooled embedding or explicit global property vector). Q3 accuracy is compared to the embeddings-only baseline to test redundancy of global information.
4. **Probing Global Properties**: Ridge regression from mean-pooled embeddings to global biophysical properties, with performance assessed via repeated K-fold CV and confidence intervals.

## Methodological Enhancements over Original Approach
### 1. Correct Handling of Data Leakage (Critique 1)
All data-dependent preprocessing (standardization of inputs, centering of targets) is performed strictly inside each cross-validation fold using only the training data. The test fold is transformed using parameters learned from the training fold. This applies to both the probing and the secondary structure experiments.

### 2. Appropriate Uncertainty Quantification (Critique 2)
We replace the unconventional bootstrap-within-CV with **repeated K-fold cross-validation**. For the probing analysis, we perform 10 repeats of 5-fold CV; the $R^2$ is computed for each fold, and a 95% confidence interval is derived from the distribution of these 50 fold-level $R^2$ values (or from the per-repeat average, depending on the desired interpretation). For the secondary structure experiments, we use 5-fold stratified cross-validation (proteins as units) repeated 5 times, and report mean Q3 and standard error across folds.

### 3. Explicit Specification of Global Property Calculations (Critiques 3 & 4)
- **Amino acid composition**: $\mathbf{y}_i^{AA}$ is the frequency vector of the 20 standard amino acids, obtained by counting and dividing by $L_i$.
- **Net charge at pH 7**: We use the Henderson–Hasselbalch equation with standard pKa values: N-terminus (pKa 9.69), C-terminus (pKa 2.34), and side chains of Asp (3.86), Glu (4.25), His (6.00), Cys (8.33), Tyr (10.07), Lys (10.53), Arg (12.48). The fractional charge of a titratable group with pKa $pK$ is $q = \frac{1}{1+10^{pH-pK}}$ for acidic groups (negative when deprotonated) and $q = \frac{10^{pH-pK}}{1+10^{pH-pK}}$ for basic groups (positive when protonated). Total charge is the sum over all residues plus the terminal backbone charges.
- **Average hydrophobicity**: The arithmetic mean of Kyte-Doolittle hydrophobicity indices (Ile: 4.5, Val: 4.2, Leu: 3.8, Phe: 2.8, Cys: 2.5, Met: 1.9, Ala: 1.8, Gly: -0.4, Trp: -0.9, Ser: -0.8, Thr: -0.7, Pro: -1.6, His: -3.2, Glu: -3.5, Gln: -3.5, Asp: -3.5, Asn: -3.5, Lys: -3.9, Arg: -4.5).
- For the 20-dimensional amino acid composition target in probing, the $R^2$ is defined as the average of the per-dimension $R^2$ values (micro-averaged variance unexplained).

### 4. Detailed Hyperparameter Tuning (Critique 5)
- **Ridge regression** (probing): The regularization strength $\lambda$ is tuned via inner 3-fold CV on the training fold, searching over a grid of $\{10^{-3}, 10^{-2}, \dots, 10^{3}\}$, using mean squared error as the criterion. The model with minimal MSE is selected.
- **Linear softmax classifier** (secondary structure): The model is logistic regression with L2 penalty; the inverse regularization strength $C$ is tuned via inner 3-fold CV on the training proteins over a logarithmic grid $\{10^{-2}, 10^{-1}, 1, 10, 10^2\}$, maximizing accuracy. Alternatively, multi-class hinge loss could be used, but softmax is preferred for probabilistic interpretation.

### 5. Special Token Handling (Critique 6)
ProtBERT tokenizes sequences with [CLS] and [SEP] tokens. We discard the embeddings corresponding to these special tokens and any padding tokens, using only the vectors from the actual amino acid residues. Mean pooling is computed exclusively over those residue embeddings.

### 6. Overfitting Control in Fusion Experiment (Critique 7)
The linear classifier in the fusion experiment is trained with the same nested CV procedure as the baseline, including inner tuning of $C$. No early stopping is needed because the model is shallow and small; instead, we rely on cross-validated parameter selection and report test performance on held-out proteins.

### 7. Direct Baseline Comparison (Critique 8)
To directly address the reproduction goal, we explicitly compare three feature sets: (a) per-residue ProtBERT embeddings, (b) hand-crafted features (one-hot amino acid + hydrophobicity + charge), (c) concatenation of both. The comparison is made within the same nested CV framework. This replaces the original method's exclusive focus on probing and fusion.

## Algorithmic Workflow
### Stage 1: Embedding and Feature Extraction (offline)
```
For each protein i:
   tokenized_seq = tokenizer(s_i, return_tensors='pt', padding=True, truncation=True)
   outputs = protbert(**tokenized_seq)
   E_i = outputs.last_hidden_state  # (1, L_max, d)
   # Remove special tokens: [CLS] at index 0, [SEP] at index corresponding to end, and any padding
   residue_mask = (tokenized_seq['input_ids'] not in {CLS_id, SEP_id, PAD_id})
   E_i_residue = E_i[residue_mask]  # shape (L_i, d)
   # Save per-residue embeddings
   # Compute hand-crafted features per residue:
   For each residue j in sequence s_i:
       f_{i,j} = [onehot(a_{i,j}); kyte_doolittle[a_{i,j}]; charge_contribution(a_{i,j})]  # dimension e.g. 22
   # Compute global properties for protein i (AA composition, net charge, average hydro) as defined.
```

### Stage 2: Secondary Structure Prediction Comparison
```
Perform 5x5 nested CV:
   For each outer repetition (1..5):
       Split proteins into 5 stratified folds (protein IDs, not residues)
       For each fold as test, rest as train:
           For each feature set in {Emb, Feat, Emb+Feat}:
               Inner 3-fold CV on train proteins to select C (L2 penalty) for logistic regression.
               Standardize features (if applicable) using training statistics.
               Train logistic regression classifier on training residue embeddings/features.
               Predict on test residues, compute Q3 accuracy.
           Store fold accuracy.
   Compute mean Q3 and 95% CI from the 25 fold-level accuracies.
```
Optionally, perform paired t-test between Emb and Feat across folds to assess significance.

### Stage 3: Fusion Experiment (Global Context)
```
For each outer fold in a 5x5 CV (same splits as stage 2):
   Create global vector g_i for each protein i, one of:
       - mean-pooled embedding z_i
       - actual global property vector y_i (compositions, charge, hydro)
       - predicted property vector ^y_i from probing model trained on train set
   For each residue j in protein i, form e_fused = [e_{i,j}; g_i]
   Train logistic regression (with inner C tuning) on train residues, evaluate Q3 on test residues.
   Compare to Emb-only baseline using per-fold differences; compute mean difference and CI.
```

### Stage 4: Probing Global Properties (Diagnostic)
```
For each target property (AA composition, charge, hydro):
   Perform 10 repeats of 5-fold CV:
       Split proteins into 5 folds (stratified)
       For each fold:
           Split current fold into train_fold, test_fold
           On train_fold:
               Standardize mean-pooled embeddings (z) to zero mean unit variance.
               Center target y (subtract mean of training values).
               Tune λ via inner 3-fold CV on train_fold using MSE.
               Train Ridge(λ) on train_fold.
           On test_fold:
               Transform z using training statistics, predict, compute R² (see definition).
       Store R² per fold.
   Report mean R² and 95% CI from the 50 fold R² values.
```

## Implementation Details
- **ProtBERT model**: Use `Rostlab/prot_bert` from Hugging Face Transformers, frozen.
- **Tokenization**: Include special tokens; ensure mask for residue embeddings by checking token IDs against tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id.
- **Logistic regression**: Use `sklearn.linear_model.LogisticRegression` with `multi_class='multinomial'`, `solver='lbfgs'`, `max_iter=1000`. Inner tuning for `C` via `GridSearchCV`.
- **Ridge regression**: Use `sklearn.linear_model.Ridge` with `solver='svd'` for efficiency. For multi-output AA composition, use `MultiOutputRegressor(Ridge())` or `Ridge` directly with appropriate shape.
- **Standardization**: `StandardScaler` fitted on training folds.
- **Random splits**: Use seeds for reproducibility; protein-level splits ensure no residue overlap.

## Complexity and Feasibility
- Embedding extraction: O(∑ L_i d) from forward pass; feasible on a single GPU.
- Per-fold training of linear classifiers: O(T d) per iteration, where T is total residues. With 25 outer folds for repeated CV, overall computation is light.
- Probing: Ridge regression on N=100, d=1024 is fast; repeated CV adds modest overhead.
- The entire pipeline can be executed on a standard workstation with GPU and 16 GB RAM within a few hours.

This refined method ensures reproducibility, statistical rigor, and direct alignment with the task of reproducing the ProtTrans finding that language model embeddings outperform hand-crafted features for secondary structure prediction.

## Research Task
复现ProtTrans论文的核心发现：使用蛋白质语言模型进行二级结构预测

## Available Data
- protein_sequences_sample.csv: 包含1000条蛋白质序列样本及其二级结构标签的CSV文件，包含字段：sequence（氨基酸序列）、secondary_structure（二级结构标签：H/E/C）、protein_id
- pretrained_embeddings.json: 预训练蛋白质语言模型（如ProtBERT）生成的嵌入向量样本，包含前100个蛋白质的嵌入表示
- protein_features.csv: 传统蛋白质序列特征：氨基酸组成、疏水性、电荷等物理化学特征

## Evaluation Criteria (Checklist)
Your work will be scored on 0 criteria:


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
