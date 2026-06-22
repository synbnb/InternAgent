Your goal is to reproduce the findings from a scientific paper by implementing the following approach.

## Reproduction Approach
We present a zero-shot method for predicting intrinsic disorder in proteins. For each residue, we compute the prediction entropy from multiple pretrained protein language models (pLMs), including ProtBERT, ProtT5, and at least one additional architecture (e.g., ProtXLNet). The entropy quantifies the model's uncertainty about the amino acid at that position given its context, reflecting the degree of structural constraint. We average these entropy scores across models to obtain a robust disorder propensity, and then apply a global threshold derived from a curated set of highly ordered reference proteins to make binary predictions. The method requires no supervised training on disorder labels and directly links self-supervised learning objectives to a biophysical property.

## Proposed Method
## Detailed Method

### 1. Notation and Setup
- Let `A = {A_1,...,A_20}` be the 20 standard amino acids.
- A protein sequence of length `L` is `S = (x_1,...,x_L)`, `x_i ∈ A`.
- We use `K` pretrained protein language models `M = {m_1,...,m_K}`. Each model `m` has a vocabulary `V_m` and a tokenizer.
- For a model `m` and position `i`, we denote the probability distribution over amino acids as `p_i^{(m)}(a)` for `a ∈ A`. This distribution is obtained by constraining the model's output to amino acid tokens and renormalizing.

### 2. Entropy Computation per Model
We describe how to compute `p_i^{(m)}(a)` for each model family, including an explicit mapping from subword tokens to amino acids.

#### Subword-to-Amino-Acid Mapping
For any model, we define for each amino acid `a` the set `T_a ⊆ V_m` of tokens that represent `a` (e.g., a single token or multiple subwords). Then:
```
p_i^{(m)}(a) = (1/Z_i) * Σ_{t ∈ T_a} q_i^{(m)}(t),
where q_i^{(m)}(t) is the model's predicted probability for token t at position i, and Z_i = Σ_{a∈A} Σ_{t∈T_a} q_i^{(m)}(t) normalizes over amino acids.
```
In practice, all pLMs used in ProtTrans (ProtBERT, ProtT5, ProtXLNet) tokenize each amino acid as a single token, so `|T_a|=1` and `Z_i=1` if the output is already confined to amino acid tokens; the mapping reduces to directly taking the token probability.

#### BERT-like Models (ProtBERT, Albert, Electra)
1. Replace the token at position `i` with the [MASK] token.
2. Feed the sequence through the model and obtain logits at position `i`.
3. Apply softmax over the vocabulary to get `q_i(t)`.
4. Compute `p_i(a)` via the mapping above.
5. Entropy: `H_i^{(m)} = -Σ_{a∈A} p_i(a) log p_i(a)`.
6. Repeat for all `i=1..L`. This requires `L` forward passes, which can be batched for efficiency.

#### T5-like Models (ProtT5)
ProtT5 uses an encoder-decoder architecture. We follow a “masked language modeling” approach adapted to its format:
1. Prepare the encoder input by replacing the token at position `i` with a unique sentinel token, e.g., `<extra_id_0>`.
2. Set the decoder input to start with the same sentinel token.
3. Perform greedy decoding for one step (or read the logits directly from the first output token position if the model supports it).
4. The model outputs a distribution over the original vocabulary. Restrict to tokens in `A` and renormalize to obtain `p_i(a)`.
5. Compute entropy as above.

#### Autoregressive Models (Transformer-XL, XLNet)
- Use a single forward pass with a causal attention mask (left-to-right ordering).
- For each position `i`, after the model processes the prefix `x_{<i}`, it produces logits for predicting the next token. Take `q_i(t)` as the probability of token `t` given the prefix.
- Apply the mapping to obtain `p_i(a) = P(x_i = a | x_{<i})`.
- Compute entropy `H_i^{(m)}`.
- Note: For XLNet, which was trained with permutation language modeling, this uses only the natural left-to-right factorization, ignoring the permutation objective. While this is an approximation, the model still captures long-range dependencies because its hidden representations integrate information from all positions via the two-stream self-attention; experiments (as in ProtTrans) show that the resulting representations are highly informative for downstream tasks. Thus, the entropy signal remains a valid indicator of disorder.

### 3. Ensemble Aggregation
We compute the disorder propensity score for residue `i` as the simple average of the per-model entropies:
```
s_i = (1/K) * Σ_{m=1}^K H_i^{(m)}
```
Averaging reduces variance and architecture-specific biases. We do not apply within-protein normalization (e.g., z-scoring), because it would break comparability with the global threshold derived from absolute entropy values (see below).

### 4. Global Threshold Determination from a Reference Set of Ordered Proteins
To make binary predictions without any disorder labels, we rely on a curated set of highly ordered protein structures. The procedure is as follows:

1. **Curate the reference set:**
   - Download all protein chains from the PDB with experimental method X-ray diffraction, resolution ≤ 2.0 Å, and no missing residues in the SEQRES records (i.e., no gaps or unknown residues).
   - Filter out chains with length < 50.
   - Remove redundancy at the 30% sequence identity level using CD-HIT.
   - The resulting set `D_ref` consists entirely of well-ordered, globular domains.

2. **Compute entropies for all residues in `D_ref`:**
   - For each sequence in `D_ref`, run the ensemble (Steps 2–3) to obtain per-residue scores `s_i`.
   - Collect all these scores into a multiset `S_ref`.

3. **Set the threshold `τ`:**
   - Let `τ = percentile(S_ref, 95)`, i.e., the 95th percentile of the score distribution in ordered proteins.
   - Rationale: In ordered regions, low entropy is expected due to strong structural constraints. By choosing the 95th percentile, we allow a 5% false positive rate (predicting disorder when the residue is actually ordered) on the reference set. This is analogous to a statistical test with significance level 0.05.

### 5. Disorder Prediction
For a new protein sequence `S`:
```
For each residue i (1..L):
    if s_i > τ  →  predicted disordered (d_i = 1)
    else        →  predicted ordered    (d_i = 0)
```

### 6. Algorithm Pseudocode
```
Algorithm ZeroShotDisorderEnsemble:
    Input: protein sequence S, set of models M, global threshold τ
    Output: binary disorder vector d[1..L]
    
    Initialize s[1..L] = 0
    For each model m in M:
        Compute H^{(m)} = [H_1,...,H_L] using model-specific procedure (Section 2)
        For i=1..L: s[i] += H_i^{(m)}
    End For
    s[i] = s[i] / K   # average over models
    For i=1..L:
        d[i] = 1 if s[i] > τ else 0
    Return d
```

### 7. Complexity Analysis
- For BERT-like and T5-like models, each position requires a forward pass. With batching of L sequences of length L, a naïve implementation incurs O(L³) time per model because the transformer’s self-attention is O(L²) per sequence and we have L sequences. In practice, L is typically ≤ 1024, and efficient batching (e.g., using diagonal masks) can reduce constant factors. Modern GPUs handle such workloads comfortably.
- Autoregressive models need only one forward pass of length L, with O(L²) time per model.
- Overall, for K models, worst-case time is O(K·L³) for masked models and O(K·L²) for autoregressive models. Memory requirements are O(L²) for attention matrices. The method is feasible on a single high-memory GPU for typical proteomes (average protein length ≈ 350).

### 8. Reproducibility and Implementation Notes
- For all models, use the pretrained checkpoints from the ProtTrans repository (e.g., `Rostlab/prot_bert`, `Rostlab/prot_t5_xl_uniref50`, `Rostlab/prot_xlnet`).
- The mapping from tokens to amino acids is provided by the tokenizer’s vocabulary file (e.g., for ProtBERT, each amino acid is a token, so T_a is the token ID for that character). Ensure that special tokens (e.g., [CLS], [SEP]) are excluded from the amino acid set.
- The reference set shall be built following the protocol in Section 4; the global threshold τ is precomputed once and reused for all predictions.
- All randomness from dropout is disabled during inference (model.eval() in PyTorch).
- Entropy computation uses natural logarithms; the threshold is consistently applied on the same scale.

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
