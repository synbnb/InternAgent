Your goal is to reproduce the findings from a scientific paper by implementing the following approach.

## Reproduction Approach
ACM-Probe is a lightweight method to predict residue-residue contacts from a single protein sequence by leveraging the internal attention patterns of the pre-trained ProtT5 language model. It learns a linear combination of symmetrized attention maps from all layers and heads, trained on a small set of protein structures. The learned weights reveal which heads specialize in local versus long-range contacts, providing interpretable insights into the model's structural knowledge without any fine-tuning. The method is easy to implement and computationally efficient, enabling zero-shot contact extraction from the frozen encoder.

## Proposed Method
1. Overview and Motivation

ProtT5 is a large protein language model pre-trained via span corruption, which masks contiguous segments of amino acids and forces the encoder to reconstruct them from left and right context. We hypothesize that this task encourages the model to learn attention patterns that correlate with inter-residue contacts, as spatially proximal residues are often the most informative for predicting missing sequence fragments. While previous studies have manually inspected individual attention heads to find contact-like behaviors, ACM-Probe provides a systematic and reproducible approach to decode contact maps from the full set of attention heads without any fine-tuning of the base model.

2. Method Details

2.1 Model and Attention Extraction
We employ the publicly available ProtT5-XL-UniRef50 model (HuggingFace identifier: Rostlab/prot_t5_xl_uniref50) in encoder-only mode (T5EncoderModel). The tokenizer is ProtTransTokenizer, identical to T5Tokenizer but with special tokens for protein sequences. For an input sequence x = (x_1, ..., x_N) of length N, the encoder produces attention matrices {A^{l,h} ∈ ℝ^{N×N}}_{l=1..L, h=1..H}, where L is the number of layers (24) and H is the number of heads (32). Each A^{l,h} is row-stochastic:  A^{l,h}_{ij} ≥ 0,  ∑_j A^{l,h}_{ij} = 1. The embeddings are not used; only the attention weights (which are accessible via output_attentions=True) are retained.

2.2 Linear Aggregation Probe
The contact prediction module is a simple linear combination of the symmetrized attention matrices from all layers and heads, followed by a sigmoid nonlinearity. For residue pair (i, j), the contact logit S_{ij} is computed as:

  S_{ij} = b + ∑_{l=1}^L ∑_{h=1}^H w^{l,h} · (A^{l,h}_{ij} + A^{l,h}_{ji}) / 2,

where w^{l,h} ∈ ℝ are learnable head-specific weights and b ∈ ℝ is a learnable bias. The predicted contact probability is P_{ij} = σ(S_{ij}), with σ(z) = 1/(1+e^{-z}). The total number of parameters is L·H + 1 = 769, making the probe extremely lightweight.

2.3 Training Procedure

Data Preparation: We use a set of protein domains with known 3D structures to train the linear probe. Specifically, we select 500 domains from the ProteinNet dataset (or a custom set derived from the PDB) with maximum sequence identity of 30% to any other domain in the set. We randomly split these into 400 training and 100 validation domains. An independent test set of 100 domains from CASP13 FM targets is reserved for final evaluation. For each domain, the ground-truth contact map M ∈ {0,1}^{N×N} is defined by Cβ distance < 8 Å, with M_{ij} = 1 if the Cβ atoms of residues i and j are within 8 Å, excluding pairs with sequence separation |i − j| ≤ 6 to avoid trivial local contacts. The list of domain identifiers, along with the exact data split, will be publicly released to ensure full reproducibility.

Hyperparameters: We train for 100 epochs with the Adam optimizer (learning rate = 1e-3, weight decay = 1e-5). The mini-batch size is 32 domains. The loss function is the average binary cross-entropy over all valid residue pairs (|i−j|>6) across all domains in the batch, plus an L2 regularization term:

  L = −(1/|I_batch|) Σ_{(i,j)∈I_batch} [ M_{ij} log P_{ij} + (1−M_{ij}) log(1−P_{ij}) ] + λ · ||w||₂²,

where I_batch is the union of valid residue pairs from all proteins in the batch, and λ = 1e-5. This formulation ensures that each residue pair contributes equally to the loss, regardless of sequence length. The per-batch averaging avoids the ambiguity of per-protein loss aggregation.

Training Loop (Pseudocode):

  Initialize w ← 0, b ← 0
  For epoch = 1 to 100:
    For batch in training_dataloader:
      batch_loss = 0; pair_count = 0
      For each protein P in batch:
        tokens = tokenize(P.sequence); input_ids, attention_mask (padding used)
        output = model(input_ids, output_attentions=True)
        attentions = output.attentions  # tuple of L tensors, each (B, H, N, N)
        # Due to padding, we extract only the valid slice for this protein:
        N = length(P.sequence)
        A_lh = [layer_attn[protein_index, :, :N, :N].detach() for layer_attn in attentions]
        # symmetrize
        A_sym = [(a + a.transpose(−2,−1)) / 2 for a in A_lh]
        # compute S for this protein
        S = b + sum(w[lh] * A_sym[lh] for lh in range(L*H))  # w flattened to (L*H,)
        P = σ(S)
        # mask out local pairs
        mask = (torch.arange(N)[:, None] - torch.arange(N)[None, :]).abs() > 6
        # also mask padding? Already handled by slicing to N.
        # accumulate loss over valid pairs
        loss_i = BCELoss(P[mask], M_i[mask])  # M_i is ground truth for protein
        batch_loss += loss_i * mask.sum()
        pair_count += mask.sum()
      avg_loss = batch_loss / pair_count + λ * ||w||₂²
      optimizer.zero_grad()
      avg_loss.backward()
      optimizer.step()

This handles variable-length sequences correctly: attention maps are sliced to the actual protein length, and the loss is aggregated by averaging over all valid pairs in the batch. The frozen ProtT5 encoder is run in evaluation mode with no gradients.

2.4 Inference
For a new protein sequence, contacts are predicted by the same procedure: tokenize, run the frozen encoder, extract and symmetrize attention maps, compute S with the trained (w,b), and apply sigmoid to obtain P. The predicted contact map can be used directly or fed into downstream structure prediction pipelines.

2.5 Analysis of Head Specialization
After training, the learned weights w provide a natural interpretability mechanism. To quantify head specialization, we perform two analyses:

a) Ablation-based importance: For each head (l,h), we set its weight to zero while keeping all others fixed, and measure the drop in precision (or F1) on the validation set. Heads whose removal causes a large performance decrease are deemed critical for contact prediction.

b) Contact range analysis: For each head, we compute the mean sequence separation of the top-K (e.g., K = 10) residue pairs with the largest symmetrized attention values, averaged over validation proteins. This reveals whether the head focuses on short-range (local secondary structure) or long-range (tertiary) contacts. We can further categorize contacts into short (|i−j| ≤ 12) and long (|i−j| > 12) and calculate the proportion of attention mass devoted to each range.

These analyses directly test the hypothesis that span corruption training induces attention heads specialized for contacts at different spatial scales. The results can be visualized as heatmaps of w^{l,h} across layers and heads, and as bar plots of contact-range distributions per head.

3. Implementation Notes
The entire pipeline can be implemented in PyTorch using the HuggingFace Transformers library. The linear probe adds negligible computational overhead—training converges in minutes on a single GPU for the 400-domain training set. Memory usage is dominated by storing attention maps: for a protein of length N, L×H×N×N floating-point values are buffered; if N exceeds 1024, one may use gradient checkpointing or process longer sequences in chunks. For typical protein lengths (N < 500), no special handling is needed.

The proposed method relies solely on the frozen ProtT5 model; no fine-tuning is required. The training set is intentionally kept small (400 domains) to demonstrate that even limited structural supervision is sufficient to unlock the contact information implicitly encoded by self-supervision. All code, trained probe weights, and data splits will be released to facilitate reproduction and further research.

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
