Your goal is to reproduce the findings from a scientific paper by implementing the following approach.

## Reproduction Approach
We present a method to evaluate whether self-supervised protein language models, specifically ProtT5, capture long-range residue coevolutionary signals without multiple sequence alignments. The core idea is to extract symmetrized self-attention scores from the final encoder layer as an estimator of residue–residue coupling strength, analogous to coevolutionary scores from MSA-based methods. By comparing these attention-derived contact maps with MSA-derived contacts (e.g., from CCMpred), we directly verify the claim that single-sequence models can learn evolutionary dependencies. The method is implemented as a lightweight probing pipeline that requires no model retraining and provides a rigorous, reproducible benchmark. A secondary linear probe on per-residue embeddings for secondary structure prediction complements the contact analysis, confirming that local structural features are also encoded.

## Proposed Method
## 1. Intuition and Theoretical Grounding
Span-corruption pretraining forces the model to reconstruct masked residues by leveraging the surrounding context. To accomplish this, the self-attention mechanism must learn to attend strongly to positions that carry complementary information about the masked region. Residue pairs that coevolve in evolutionary processes exhibit correlated substitution patterns; therefore, when one residue is masked, attending to its coevolving partner provides essential clues for reconstruction. Consequently, the self-attention weights can be interpreted as a learned pairwise coupling score that mirrors coevolutionary potentials derived from MSAs. Unlike prior work that trains separate classifiers on embeddings, we directly use the raw attention weights as a hypothesis-free probe of residue–residue dependencies.

## 2. Attention-Based Contact Score Extraction
Given a protein sequence \( x = (x_1, \dots, x_L) \), we feed it through the pretrained ProtT5 encoder only (decoder is not used). The model tokenizes each residue character separated by spaces, respecting the T5 vocabulary. For sequences longer than the maximum position embedding (1024), we truncate to the first 1024 residues.

Let \( A^{(l,h)} \in \mathbb{R}^{L \times L} \) be the self-attention weight matrix from layer \( l \) and head \( h \), where \( A^{(l,h)}_{ij} \) is the attention from position \( i \) to \( j \). We focus on the final layer (\( l = L_f \)) as it typically captures the most global dependencies. For each head, we symmetrize the attention: \( \tilde{A}^{(h)}_{ij} = (A^{(L_f,h)}_{ij} + A^{(L_f,h)}_{ji}) / 2 \). The per-head maps are then averaged across all \( H \) heads:

\[
S_{ij} = \frac{1}{H} \sum_{h=1}^{H} \tilde{A}^{(h)}_{ij}
\]

Optionally, to reduce background biases common in contact prediction, we apply the Average Product Correction (APC): \( S^{\text{APC}}_{ij} = S_{ij} - \frac{\sum_{k} S_{ik} \sum_{k} S_{kj}}{\sum_{kl} S_{kl}} . \) The resulting \( S^{\text{APC}} \) (or \( S \)) is taken as the predicted contact score for the pair \((i,j)\).

## 3. Secondary Structure Probing (Supporting Analysis)
Although the main innovation lies in contact mining, we also probe per-residue embeddings for secondary structure to demonstrate that local structural features are encoded. Using the final hidden states \( h_i \) from the ProtT5 encoder, a simple linear classifier with softmax predicts the secondary structure label (3- or 8-state) for each residue: \( \hat{y}^{SS}_i = \text{softmax}(W_{SS} h_i + b_{SS}) \). Training is done with cross-entropy loss on annotated residues.

## 4. Implementation Protocol
### 4.1 Datasets
- **Contact validation**: A non-redundant set of protein chains from the Protein Data Bank (PDB) with resolution ≤ 2.5 Å, length between 50 and 1024, and release date before May 2018 are collected. Redundancy is removed at 40% sequence identity using MMseqs2. Overall, we curate ~4000 chains, split into 80% training, 10% validation, and 10% test sets. Ground-truth contacts are defined as residue pairs with Cβ–Cβ distance < 8 Å and sequence separation |i − j| ≥ 6 (or ≥ 24 for long-range contact evaluation).
- **Secondary structure**: Standard datasets such as CB513 or CASP12 targets are used, with splits as originally defined. DSSP annotations provide the 3-state labels (H, E, C) or 8-state labels.

### 4.2 Model and Extraction
- ProtT5-XL-U50 is loaded from HuggingFace (`Rostlab/prot_t5_xl_uniref50`).
- Tokenization: sequence characters are space-joined and encoded; special tokens are added according to T5 format.
- A single forward pass is performed with `output_attentions=True`. No gradient computation is needed.
- Attention matrices from the last encoder layer are extracted. Per-residue hidden states are also obtained.

### 4.3 Contact Probe Evaluation
- For each protein in the test set, compute \( S^{\text{APC}} \).
- Predict the top \( L/5 \) long-range (|i−j| ≥ 24) or top \( L/2 \) medium+long-range (|i−j| ≥ 6) pairs as contacts.
- Performance is measured by precision (percentage of predicted contacts that are true contacts), recall, and F1-score.
- Baselines:
  - **MSA-based**: Jackhmmer is used to search UniRef90 with 3 iterations to build an MSA; CCMpred then infers a Markov random field and outputs coupling scores. The top \( L/5 \) pairs are taken as predicted contacts.
  - **Random**: A random ordering of pairs serves as a lower bound.
  - **ProtBERT**: Run the same attention extraction on a ProtBERT model fine-tuned on the same sequence set (or the publicly available Rostlab/ProtBert) to compare objectives.

### 4.4 Secondary Structure Probe Training
- A linear layer with input dimension 1024 (ProtT5 hidden size) and output dimension 3 (or 8) is trained on the training set.
- Optimizer: Adam with learning rate 1e-3, batch size 32, maximum 50 epochs. Early stopping is applied with patience 5 on validation loss.
- Evaluation: per-residue Q3 accuracy on the test set.

### 4.5 Reproducibility
All code is written in Python using PyTorch and HuggingFace Transformers. Random seeds are fixed. Data splits and preprocessing steps are documented. The entire pipeline runs on a single GPU (e.g., NVIDIA V100) within a few hours for the full dataset.

## 5. Algorithmic Pseudocode
```
Algorithm: AttentionCoEvoProbe
Input: Test sequences {x}, pretrained ProtT5, ground-truth contacts
Output: Precision, Recall, F1 for predicted contacts

1. Load tokenizer and model.
2. For each sequence x:
   a. tokens = tokenizer(' '.join(list(x)), max_length=1026, truncation=True)
   b. outputs = model(tokens['input_ids'], output_attentions=True)
   c. attentions = output.attentions[-1]  # shape (layers, batch, heads, L, L)
   d. S = average_and_symmetrize(attentions)  # shape (L, L)
   e. S_apc = apply_apc(S)
   f. Store S_apc for evaluation.

3. For each protein, select top predicted pairs:
   - Filter by separation (e.g., |i-j| ≥ 24)
   - Sort pairs by decreasing S_apc, take top L/5.

4. Compute precision, recall, F1 against true contacts.
5. (Optional) Compare with MSA-based CCMpred results.
```

## 6. Validation and Interpretation
We expect that the attention-based contact scores correlate strongly with both MSA-derived couplings and experimental contacts. A high precision for long-range contacts would confirm that the model indeed captures coevolutionary dependencies. Ablations (ProtBERT, random attention) are used to demonstrate that the signal is not purely due to the Transformer architecture but is induced by the span-corruption objective. This provides a mechanistic validation of the paper’s central claim without relying on opaque classifiers or theoretical contrivances.

## Research Task
复现论文《ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning》的核心发现

## Available Data
- UniRef: UniRef数据库，用于训练蛋白质语言模型的无标注蛋白质序列数据集。
- BFD (Big Fantastic Database): BFD数据库，用于训练蛋白质语言模型的无标注蛋白质序列数据集。

## Evaluation Criteria (Checklist)
Your work will be scored on 5 criteria:
  Item 0 (type=text, weight=0.25): 模型架构与训练规模：评估研究是否采用了多种自回归和自编码模型架构（Transformer-XL, XLNet, BERT, Albert, Electra, T5），并在大规模计算资源（Summit超级计算机5616个GPU和TPU Pod最多1024个核心）上进行训练，以充分探索蛋白质语言模型的潜力。
  Item 1 (type=text, weight=0.20): 数据集覆盖与多样性：评估研究是否使用了大规模且多样化的无标注蛋白质序列数据集（UniRef和BFD），以确保模型能够学习到广泛的生物物理特征。
  Item 2 (type=text, weight=0.20): 无标注数据特征提取能力：评估研究是否验证了无标注数据的原始pLM嵌入能够捕获蛋白质序列的生物物理特征，而无需依赖人工标注。
  Item 3 (type=text, weight=0.20): 下游任务性能突破：评估研究是否在蛋白质二级结构预测中，使用最具信息量的嵌入（ProtT5）首次在不使用多序列比对（MSA）或进化信息的情况下，超越了最先进的方法。
  Item 4 (type=text, weight=0.15): 自监督学习框架的有效性：评估研究是否通过自监督学习（自回归和自编码模型）从无标注蛋白质序列中学习到可迁移的特征，并验证其在下游任务（如亚细胞定位）中的通用性。

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
