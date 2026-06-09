Your goal is to reproduce the findings from a scientific paper by implementing the following approach.

## Reproduction Approach
This method improves secondary structure prediction from frozen ProtBERT embeddings by jointly training a linear classifier and a contrastive auxiliary loss. A lightweight two-layer MLP with a residual connection, initialized near identity, serves as a shared projection head for both tasks. Positive pairs for the contrastive loss are efficiently constructed using precomputed local context embeddings: for each anchor residue, a positive is sampled from those with cosine similarity above a fixed threshold (derived from the top 20% most similar context pairs). By sharing the projection head, the contrastive signal directly refines the representations used for classification. When the contrastive loss weight is zero, the residual scaling is set to zero, exactly recovering the baseline linear probe. This framework is tailored to frozen language models and ensures minimal added parameters and computational overhead.

## Proposed Method
## Method Description

### Overview
The proposed method, Shared Projection Contrastive Learning (SPCL), augments a standard linear probe on frozen ProtBERT embeddings with a contrastive auxiliary loss. Both tasks share a lightweight projection head $f_\theta$, a two‑layer MLP with a residual connection initialized near identity. The contrastive loss uses positive pairs built from residues whose local context embeddings (averaged over a $\pm 5$ residue window) exhibit high cosine similarity, as computed from the frozen ProtBERT. A fixed similarity threshold, precomputed once on a large subsample, selects the top 20% most similar pairs as candidate positives. The total loss is $\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{contr}}$, with $\lambda \ge 0$. Setting $\lambda = 0$ reverts the projection head to an identity mapping, exactly recovering the baseline linear probe.

### Detailed Components

#### Frozen Embeddings and Local Context
Per‑residue ProtBERT embeddings $\mathbf{e}_t \in \mathbb{R}^d$ ($d=1024$) are precomputed and kept fixed. For each residue $t$, a local context embedding is obtained by averaging its $\pm w$ neighborhood (here $w=5$):
$$\mathbf{c}_t = \frac{1}{2w+1} \sum_{k=-w}^{w} \mathbf{e}_{t+k},$$
with padding handled by truncating at sequence boundaries. These context embeddings capture the immediate sequence environment and are also precomputed once.

#### Shared Projection Head
A residual projection head $f_\theta$ processes the frozen embedding $\mathbf{e}_t$ to produce a refined representation $\mathbf{z}_t$:
$$\mathbf{z}_t = \mathbf{e}_t + \alpha \cdot g_\theta(\mathbf{e}_t),$$
where $g_\theta$ is a two‑layer MLP:
$$g_\theta(\mathbf{e}) = \mathbf{W}_2\,\text{ReLU}(\mathbf{W}_1 \mathbf{e} + \mathbf{b}_1) + \mathbf{b}_2,$$
with $\mathbf{W}_1 \in \mathbb{R}^{h \times d}$, $\mathbf{W}_2 \in \mathbb{R}^{d \times h}$, $\mathbf{b}_1 \in \mathbb{R}^h$, $\mathbf{b}_2 \in \mathbb{R}^d$, and $h \ll d$ (e.g., $h=128$). The factor $\alpha$ controls the residual contribution: $\alpha=1$ when $\lambda>0$, and $\alpha=0$ when $\lambda=0$.  
**Initialization:** To ensure the residual branch initially contributes almost nothing, we set $\mathbf{b}_2 = \mathbf{0}$ and initialize $\mathbf{W}_2$ with small values (e.g., $\mathcal{N}(0, 0.01)$). $\mathbf{W}_1$ and $\mathbf{b}_1$ use standard He initialization; all biases in $g_\theta$ except $\mathbf{b}_2$ are zero‑initialized.

#### Linear Classifier
A single linear layer maps $\mathbf{z}_t$ to 3‑state secondary structure logits:
$$\hat{y}_t = \text{softmax}(\mathbf{W} \mathbf{z}_t + \mathbf{b}), \quad \mathbf{W} \in \mathbb{R}^{3 \times d},\; \mathbf{b} \in \mathbb{R}^3.$$
The classification loss is the standard cross‑entropy over a batch:
$$\mathcal{L}_{\text{cls}} = \frac{1}{|\mathcal{B}|}\sum_{t\in\mathcal{B}} \text{CE}(y_t, \mathbf{W}\mathbf{z}_t+\mathbf{b}).$$

#### Contrastive Positive Pair Selection
A fixed similarity threshold $s^*$ is precomputed once to identify residue pairs with highly similar local contexts.
- Sample a large subset $\mathcal{S}$ of $M$ residues (e.g., $M=10^5$).
- Compute all pairwise cosine similarities $\text{sim}_{ij} = \langle \mathbf{c}_i, \mathbf{c}_j \rangle / (\|\mathbf{c}_i\|\|\mathbf{c}_j\|)$ for $i,j \in \mathcal{S}$.
- Set $s^*$ to the $80^{th}$ percentile of these similarities (i.e., the threshold above which only 20% of pairs lie). This value is stored and reused throughout training.  
During batch processing, for each anchor $a$, the set of candidate positives within the batch is $\mathcal{P}_a = \{ b \in \mathcal{B} \setminus \{a\} \mid \text{sim}(\mathbf{c}_a, \mathbf{c}_b) \ge s^*\}$. If $\mathcal{P}_a$ is non‑empty, a positive $p(a)$ is sampled uniformly; otherwise the anchor is ignored in the contrastive loss.

#### Contrastive Loss
Using the InfoNCE formulation with temperature $\tau$ (e.g., $\tau=0.1$), the contrastive loss over the set of valid anchors $\mathcal{A} = \{a \in \mathcal{B} \mid \mathcal{P}_a \neq \emptyset\}$ is
$$\mathcal{L}_{\text{contr}} = -\frac{1}{|\mathcal{A}|}\sum_{a\in\mathcal{A}} \log \frac{\exp\bigl(\cos(\mathbf{z}_a, \mathbf{z}_{p(a)})/\tau\bigr)}{\sum_{b\in\mathcal{B} \setminus \{a\}} \exp\bigl(\cos(\mathbf{z}_a, \mathbf{z}_b)/\tau\bigr)}.$$
Crucially, the anchor $a$ is excluded from the denominator to avoid a trivial self‑similarity term.

#### Joint Training Objective
The full loss for a batch is $\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{contr}}$, where $\lambda$ is a scalar weighting the auxiliary task.

### Algorithmic Workflow

**Precomputation stage (once):**
1. Compute frozen per‑residue embeddings $\mathbf{e}_t$ for all proteins using the pretrained ProtBERT.
2. For each residue, compute the local context embedding $\mathbf{c}_t$.
3. Sample a subset $\mathcal{S}$ (e.g., $10^5$ residues) and determine the fixed similarity threshold $s^*$ as the $80^{th}$ percentile of pairwise cosine similarities of their $\mathbf{c}$ vectors.

**Training loop:**
```
Input: precomputed e_t, c_t, labels y_t, s*, λ, τ, h
Initialize θ (projection head) and W,b (classifier) as described.
For each epoch:
    For each batch B (flat list of residues from multiple proteins):
        z_t = e_t + α * g_θ(e_t)   // α = 1 if λ>0 else 0
        L_cls = mean( CE(softmax(W z_t + b), y_t) )
        If λ > 0:
            Using precomputed c_t in B, determine anchors A and positive partners p(a) as described.
            Compute L_contr via InfoNCE (excluding anchor from denominator).
            L = L_cls + λ * L_contr
        Else:
            L = L_cls
        Backpropagate L and update θ, W, b
```

**Batching strategy:** Sequences of similar lengths are grouped into minibatches, padded to the maximum length in the batch, and then flattened into a list of residue tokens. The precomputed $\mathbf{c}_t$ are loaded accordingly. This ensures that the local context averaging remains consistent and avoids cross‑protein boundary issues because $\mathbf{c}_t$ were computed on complete sequences.

**Complexity:** With precomputed $\mathbf{c}_t$, the threshold estimation is a one‑time $O(M^2 d)$ operation ($M=10^5$ yields $\approx 10^{10}$ operations, manageable on a GPU). Training complexity per batch is $O(B d h + B^2 d)$, dominated by $B=256$ and $d=1024$, which fits comfortably in GPU memory.

### Theoretical Grounding
The InfoNCE loss maximizes a lower bound on the mutual information $I(\mathbf{z}; \text{context})$ (Oord et al., 2018). By forcing the projection head to produce similar $\mathbf{z}$ for residues whose frozen context embeddings are already similar, we encourage the refined representation to preserve and amplify the local structural signals captured by ProtBERT. This acts as a regularizer that clusters residues with analogous secondary structure environments, thereby improving the linear separability of the classifier.

### Key Advantages
- **Exact baseline recovery:** $\alpha=0$ when $\lambda=0$ guarantees the linear probe is reproduced without any contamination.
- **Efficient positive mining:** a single precomputed threshold replaces expensive per‑epoch similarity searches.
- **Minimal overhead:** the projection head adds only $\approx 2.6\times10^5$ parameters.
- **Solid theoretical foundation:** grounded in mutual information maximization and local context similarity.

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
