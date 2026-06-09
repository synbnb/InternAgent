Your goal is to reproduce the findings from a scientific paper by implementing the following approach.

## Reproduction Approach
This method introduces a unified, statistically rigorous framework for quantifying how per-residue ProtBERT embeddings linearly encode local physicochemical properties critical for protein structure. Instead of training separate probes for each property, we design a joint multi‑task linear model that simultaneously predicts hydrophobicity, charge, and amino acid identity from the embedding vector using a shared low‑rank transformation. This approach eliminates the averaging of incompatible per‑property metrics, provides a single, interpretable goodness‑of‑fit (cross‑validated R²), and directly tests the hypothesis that embeddings linearly capture the biophysical information needed for secondary structure. We then train a linear logistic regression classifier on the same embeddings for three‑state secondary structure prediction and compare its accuracy with a model that additionally receives explicit residue‑level features. A paired Wilcoxon signed‑rank test is used to assess whether explicit features provide any advantage. The joint probe not only improves reproducibility and statistical soundness but also serves as a principled diagnostic for downstream task sufficiency.

## Proposed Method
## 1. Data Preparation and Embedding Extraction
 
**Input Data:** A set of N proteins, each given by its amino acid sequence, per‑residue secondary structure labels (H/E/C), and a pretrained ProtBERT model (e.g., `Rostlab/prot_bert` from HuggingFace).
 
**Per‑Residue Embeddings:**  
- For each protein sequence, pass it through ProtBERT with a truncation length of at most 512 residues (longer sequences are split into overlapping chunks of 512 with overlap 10, and per‑residue embeddings from overlapping regions are averaged).  
- Extract the hidden states from the **last transformer layer** (output of the final encoder block) of shape (sequence_length, 1024).  
- Remove the special tokens: discard the vectors corresponding to `[CLS]` (position 0) and `[SEP]` (position L+1). The per‑residue embeddings are **not L2‑normalized**; raw magnitudes are preserved to avoid distorting linear relationships.  
- Store alignment: residue j in the original sequence corresponds to embedding vector `x_{i,j} ∈ R^1024`.
 
## 2. Physicochemical Property Computation
 
For each residue, we define three property targets:
 
**Hydrophobicity:**  
Table of Kyte‑Doolittle hydrophobicity indices (precise values):
 
| Amino Acid | Hydrophobicity |
|------------|----------------|
| A          | 1.8            |
| R          | -4.5           |
| N          | -3.5           |
| D          | -3.5           |
| C          | 2.5            |
| Q          | -3.5           |
| E          | -3.5           |
| G          | -0.4           |
| H          | -3.2           |
| I          | 4.5            |
| L          | 3.8            |
| K          | -3.9           |
| M          | 1.9            |
| F          | 2.8            |
| P          | -1.6           |
| S          | -0.8           |
| T          | -0.7           |
| W          | -0.9           |
| Y          | -1.3           |
| V          | 4.2            |
 
The value is assigned as a scalar `h_{i,j}`.
 
**Charge at pH 7:**  
- Asp (D), Glu (E): -1  
- Lys (K), Arg (R): +1  
- His (H): +0.1 (average charged fraction at pH 7)  
- All others: 0  
Scalar `c_{i,j}`.
 
**Amino Acid Identity:**  
A 20‑dimensional one‑hot vector `a_{i,j} ∈ {0,1}^20`, where the k‑th entry is 1 if the residue is amino acid k (in alphabetical order: A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V).
 
## 3. Joint Multi‑Task Linear Probing
 
We build one linear model that simultaneously maps each embedding `x_{i,j}` to all property dimensions. Let the full target matrix `Y` of size `T × 22` have rows `y_{i,j} = [h_{i,j}, c_{i,j}, a_{i,j}]`, i.e., concatenating the hydrophobicity scalar, charge scalar, and the 20‑dimensional one‑hot vector. The model is:
 
```
Ŷ = X W + 1 b^T,   W ∈ R^{1024×22}, b ∈ R^{22}
```
 
To impose a shared structure and regularise, we factor `W = P Q^T`, with `P ∈ R^{1024×r}`, `Q ∈ R^{22×r}`, where `r ≪ min(1024,22)` is a small rank (set `r = 5`). This forces properties to be predicted from a common low‑dimensional linear transformation of the embedding. The loss function is multi‑output ridge regression:
 
```
L(P,Q,b) = ||Y - X P Q^T - 1 b^T||_F^2 + λ (||P||_F^2 + ||Q||_F^2)
```
 
The parameters are estimated via alternating least squares or directly by solving the full ridge problem on the stacked weight matrix with a nuclear‑norm penalty? To keep implementation simple, we avoid explicit low‑rank constraint in optimization and instead solve a standard multi‑output ridge:
 
```
min_{W,b} ||Y - XW - 1b^T||_F^2 + λ ||W||_F^2
```
 
and then, if desired, compute a low‑rank approximation of the learned W to analyse shared components. For the purpose of probing, the full‑rank multi‑output ridge is sufficient and directly yields a coefficient of determination.
 
**Cross‑validation and R² metric:**  
We perform 5‑fold cross‑validation at the protein level (folds are random partitions of the set of protein IDs, ensuring no residue from the same protein appears in both train and validation). Let `V_k` be the validation fold for fold k. 
 
For each fold, train ridge on the training folds, selecting λ from the set `{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 10^2}` via inner 3‑fold CV on the training set (splitting again by protein). The metric for inner selection is the overall R² on the inner validation set:
 
```
R²_total = 1 - (sum over residues in validation of ||y - ŷ||^2) / (sum over residues of ||y - ȳ||^2)
```
 
where ȳ is the mean target vector across all validation residues. After selecting λ, retrain on the full training folds and evaluate on `V_k` to obtain `R²_k`. The final probing score is the average over folds: `R²_cv = (1/5) Σ R²_k`. A high `R²_cv` (e.g., >0.7) indicates strong linear encoding of the combined properties.
 
As a supplementary measure, we also compute per‑property R² from the same trained model (using the corresponding components of y), but the primary metric is the joint R², which circumvents the averaging issues of separate probes.
 
## 4. Secondary Structure Classification
 
We train two multinomial logistic regression classifiers at the residue level:
 
**Model A (embedding‑only):** Input is `x_{i,j} ∈ R^{1024}`.  
**Model B (augmented):** Input is concatenation `[x_{i,j}; f_{i,j}]` where `f_{i,j} = [h_{i,j}, c_{i,j}] ∈ R^2` (scalar hydrophobicity and charge).  
 
Both models share the form:
 
```
P(y = c | z) = exp(w_c^T z + b_c) / Σ_{c'} exp(w_{c'}^T z + b_{c'})
```
 
with `c ∈ {H,E,C}` (encoded as 1,2,3). Loss is cross‑entropy with L2 penalty:
 
```
L_cls = - Σ_{i,j} log P(y_{i,j} | z_{i,j}) + λ1 Σ_c ||w_c||^2
```
 
**10‑fold cross‑validation:** Proteins are split into 10 folds. For each fold, train on 9 folds, select `λ1` via inner 3‑fold CV on the training proteins (grid: `{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10}`), using Q3 accuracy on the inner validation set as selection criterion. Retrain on all 9 folds with best λ1, evaluate on hold‑out fold to obtain per‑protein Q3 accuracy. The Q3 for protein i is:
 
```
Q_{3,i} = (1/L_i) Σ_{j=1}^{L_i} 1[ŷ_{i,j} = y_{i,j}]
```
 
Aggregate per‑protein Q3 values across all 10 folds for both models.
 
**Statistical comparison:** Compute paired differences `Δ_i = Q_{3,i}^{augmented} - Q_{3,i}^{embedding}` for each protein (those appearing in the same fold). Perform a two‑sided Wilcoxon signed‑rank test. A p‑value > 0.05 indicates no significant improvement from explicit features, supporting the hypothesis that embeddings already contain the necessary biophysical information.
 
## 5. Algorithmic Workflow (Pseudocode)
 
```
Algorithm: Joint Multitask-Probe Structure Prediction
Input: Protein sequences with SS labels, ProtBERT model
Output: Cross-validated R² from joint probe; per‑protein Q3 for both classifiers; Wilcoxon p‑value
 
1. For each protein:
   a. Embedding: Obtain per-residue vectors from ProtBERT (last layer, remove [CLS]/[SEP], no normalization)
   b. Target matrix Y: compile row for each residue = [hydro, charge, one‑hot(aa)]
2. Joint probing (5‑fold):
   Split proteins into 5 folds.
   For each fold:
     Train multi‑output ridge: Ŷ = XW + 1b^T, select λ by 3‑fold inner CV using joint R².
     Compute joint R² on validation fold.
   Average R² across folds → R²_cv
3. Classification (10‑fold):
   Split proteins into 10 folds.
   For each fold:
     Train model A (embedding‑only logistic) with inner CV to select λ1_A.
     Train model B (augmented logistic) with inner CV to select λ1_B.
     For each protein in test fold, compute Q3 under both models.
   Collect per‑protein Q3_A, Q3_B.
4. Comparison:
   Compute Δ_i = Q3_B(i) - Q3_A(i) for proteins (matching by ID across folds).
   Wilcoxon signed‑rank test → p‑value.
5. Return R²_cv, mean Q3_A, mean Q3_B, p‑value.
```
 
## 6. Implementation Feasibility
 
All components use standard numerical libraries (scikit‑learn: `RidgeCV` for multi‑output ridge, `LogisticRegressionCV` for multinomial classification, `cross_val_score`, `Wilcoxon` from `scipy.stats`). Embeddings are obtained via `transformers` library. Physicochemical tables are hard‑coded. Complexity is dominated by the `X^T X` calculation (O(T d²) with d=1024). For 1000 proteins averaging 300 residues, T≈3e5, the total computation is under one minute on a modern CPU. The detailed hyperparameter grids and inner CV procedures ensure exact reproducibility.
 
## 7. Innovation Highlights
 
The method departs from traditional separate probing by introducing a **single joint multi‑task linear probe** that predicts all relevant properties simultaneously from the embedding. This eliminates the flawed practice of averaging R² across binary regressions and provides a holistic, interpretable measure (`R²_cv`) of linear encodability. The subsequent classification comparison and statistical test directly link probing quality to downstream utility, offering a reproducible, end‑to‑end pipeline for validating the biophysical richness of protein language model representations.

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
