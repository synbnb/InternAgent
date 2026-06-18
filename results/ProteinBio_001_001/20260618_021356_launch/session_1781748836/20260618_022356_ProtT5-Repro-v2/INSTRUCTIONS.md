Your goal is to reproduce the findings from a scientific paper by implementing the following approach.

## Reproduction Approach
This method provides a fully corrected and precisely specified pipeline to reproduce the ProtT5 result: training a T5-based denoising autoencoder on UniRef50 and BFD, and using frozen embeddings to achieve state‑of‑the‑art secondary structure prediction without multiple sequence alignments. Key enhancements include a mathematically explicit span corruption procedure that guarantees a fixed 15% token corruption budget and avoids cross‑sequence contamination through per‑sequence corruption followed by attention‑masked packing.

## Proposed Method
We detail the corrected end‑to‑end protocol, comprising data preprocessing, self‑supervised pretraining with denoising, embedding extraction, and linear evaluation.

Vocabulary
Let ℰ = {A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y} be the 20 standard amino acids, and ℬ = {B, Z, J, O, U, X} the ambiguous codes (asparagine/aspartic acid, glutamine/glutamic acid, leucine/isoleucine, pyrrolysine, selenocysteine, unknown). The full amino‑acid vocabulary is 𝒜 = ℰ ∪ ℬ (size 25). Special tokens: ⟨pad⟩, ⟨eos⟩, and sentinel tokens ⟨extra_id_i⟩ for i = 0,…,99. The model vocabulary is 𝒱 = 𝒜 ∪ {⟨pad⟩,⟨eos⟩} ∪ {⟨extra_id_i⟩}_{i=0}^{99}.

Span Corruption (Corrected)
For a protein sequence 𝐱 = (x₁,…,x_T) with x_t ∈ 𝒜, we corrupt exactly 15 % of its tokens as follows:
1. Repeat until the total number of tokens designated for corruption reaches ⌊0.15·T⌋:
   - Draw a span length ℓ from a Poisson distribution with mean 3, clamped to [2, 10] and to the remaining token budget.
   - Choose a random start position such that the span lies entirely within the sequence and does not overlap previously selected spans.
   - The span tokens are added to the corruption set.
2. Sort the selected spans by start position. Let there be K spans S₁,…,S_K.
3. Replace each span S_k in the sequence with the sentinel token ⟨extra_id_{k-1}⟩, producing the corrupted input 𝐱^cor.
4. The target sequence 𝐲 is defined as:
   𝐲 = ⟨extra_id_0⟩, S₁ tokens, ⟨extra_id_1⟩, S₂ tokens, …, ⟨extra_id_{K-1}⟩, S_K tokens, ⟨extra_id_K⟩, ⟨eos⟩.
During training, the model receives 𝐱^cor and must generate 𝐲 autoregressively, minimizing token‑wise cross‑entropy.

Model Architecture
We use the T5‑3B configuration:
- Encoder layers: 24; Decoder layers: 24.
- Model dimension: d_model = 1024.
- Feed‑forward dimension: d_ff = 16384.
- Attention heads: n_heads = 128, with per‑head dimension d_v = d_k = 8.
- Relative position embeddings (bucketed up to 512).
- Vocabulary size: |𝒱| = 25 + 2 + 100 = 127. (25 amino acids, pad, eos, 100 sentinels)
- Total parameters ≈ 3 billion.

Data Preprocessing (Leakage‑Free)
1. Download UniRef50 (release 2021_03) and BFD (latest). Filter: discard sequences with length < 50 or with >20 % of residues outside ℰ (i.e., in ℬ).
2. Tokenize each sequence into IDs from 𝒜.
3. **Per‑sequence corruption**: For every accepted sequence, apply the span corruption procedure described above. Store the corrupted sequence and its target.
4. **Packing into blocks** of maximum length 512 tokens:
   - Concatenate corrupted sequences, separating them with the ⟨eos⟩ token, until adding the next sequence would exceed 512 tokens. Then pad with ⟨pad⟩ to 512.
   - Correspondingly pack the targets: for each corrupted sequence, take its target tokens and insert an ⟨eos⟩ separator between targets of different sequences. Pad to a uniform length (e.g., 512) with ⟨pad⟩.
   - Record the length of each original example inside the block (for attention masking).
5. Create attention masks that restrict self‑attention to tokens belonging to the same original example (i.e., between two ⟨eos⟩ separators). This ensures no cross‑sequence information is used.

Pretraining
- Optimizer: AdamW with β₁=0.9, β₂=0.999, ε=1e-8, weight decay 1e-5.
- Learning rate schedule: Inverse square root with peak 2e-4, warmup over 10k steps.
- Batch size: 128 blocks per device; gradient accumulation to effective batch size 2048.
- Distributed training on 512 TPU v3 cores (or equivalent), model parallelism as needed.
- Train for 1 M steps (~100 epochs over the packed sequences).
- Monitor perplexity on a held‑out set of 10 k sequences.
- Model initialization: All weight matrices are initialised with a normal distribution of mean 0 and std d_model^{−0.5}; embeddings with a truncated normal of std 1e-2; biases to zero.

Embedding Extraction
For a protein of length L, feed the clean (uncorrupted) sequence through the encoder. The last‑layer hidden states H = (𝐡₁,…,𝐡_L), 𝐡_i ∈ ℝ^1024, are taken as per‑residue embeddings.

Secondary Structure Evaluation
- Training/validation: Use the NetSurfP‑2.0 training dataset (TS115). Set aside 10 % randomly as validation.
- Test: CB513, filtered to remove sequences with >25 % identity to any sequence in TS115 (standard PISCES filtering).
- Input features: For each residue, concatenate its embedding 𝐡_i with a 25‑dimensional one‑hot encoding of the amino acid (ambiguous residues map to their own bin). Resulting dimension = 1024 + 25 = 1049.
- Classifier: A single linear layer (1049 → 3) with softmax.
- Train with cross‑entropy loss, Adam (lr=1e-3), batch size 64, for up to 50 epochs with early stopping (patience 5 on validation loss).
- Report Q3 accuracy on CB513.

Theoretical Basis
The denoising objective forces the model to learn the conditional distribution p(x_{missing} | x_{context}), which captures inter‑residue dependencies such as hydrogen bonding and hydrophobic packing. Because the spans vary in length and position, the model must integrate both short‑range (local secondary structure) and long‑range (tertiary contacts) context. Training on unlabelled data at scale allows the model to internalise the constraints of physical protein folding without any structural supervision.

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
