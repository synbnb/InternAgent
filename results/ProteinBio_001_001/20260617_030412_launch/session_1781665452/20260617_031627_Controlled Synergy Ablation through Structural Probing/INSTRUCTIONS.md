Your goal is to reproduce the findings from a scientific paper by implementing the following approach.

## Reproduction Approach
We propose a systematic experimental framework to dissect the contributions of pre-training objective, model architecture, and data scale to the performance of protein language models, with a focus on understanding the emergence of co-evolutionary features in embeddings. The study includes multiple architectures from the ProtTrans suite (BERT, Albert, T5, etc.) and introduces custom variants to isolate specific factors. A key innovation is a novel metric, Structural Attention Precision (SAP), which quantifies the degree to which a model's attention maps recover experimentally determined residue–residue contacts, providing a direct measure of structural learning without requiring multiple sequence alignments. By combining controlled ablation with structural probing, we provide causal evidence for the synergistic effect of span-denoising, bidirectional context, and data scale in achieving state-of-the-art single-sequence secondary structure prediction.

## Proposed Method
The experimental protocol consists of four stages: (I) Model definition, (II) Pre-training under uniform conditions, (III) Downstream evaluation with linear probes, and (IV) Structural knowledge quantification via the SAP metric.

**Stage I: Model Architecture and Variant Design**

We consider two tiers of models to satisfy the scope of ProtTrans reproduction and targeted ablation:

*Tier 1 – Broad Architecture Comparison (for benchmarking against ProtTrans findings)*

- BERT (encoder-only, masked language model)
- Albert (parameter-efficient encoder-only, masked language model)
- Electra (encoder-only, replaced token detection)
- XLNet (generalized autoregressive pretraining)
- T5 (encoder-decoder, span-denoising)

All are implemented with comparable total parameters (~100M) and trained on the same data (UniRef50 + BFD) for 10B tokens.

*Tier 2 – T5-Specific Ablation Variants (to isolate synergistic factors)*

We focus on the T5 architecture because ProtT5 (a T5 variant) achieved the strongest single-sequence secondary structure results. All variants below maintain the same encoder-decoder backbone and number of parameters (~100M) unless otherwise noted.

1. **Span-Denoising T5 (SD-T5)**: The baseline. Input sequences are corrupted by replacing random contiguous spans of 2–5 residues with sentinel tokens (e.g., <extra_id_0>). The model is trained to reconstruct the original spans as target text, using a standard cross-entropy loss.
2. **Masked Language Model T5 (MLM-T5)**: Same architecture, but instead of span corruption, we randomly mask 15% of individual residues (replaced with <mask>) and train the model to predict only those masked positions. This isolates the effect of pre-training objective (span-denoising vs. single-residue masking) while keeping the encoder-decoder structure constant.
3. **Autoregressive T5 (AR-T5)**: A decoder-only model (encoder is removed) that receives a corrupted input sequence with sentinel placeholders, e.g., "M K G [S1] L A [S2] D " (where [S*] are special tokens indicating missing spans). The model is trained to predict the original residues following each sentinel autoregressively, i.e., given the prefix, it generates the sequence corresponding to the first missing span (e.g., "V I N") until the next sentinel. This isolates the role of bidirectional context in the baseline.
4. **Data-Scale Variants**: SD-T5 trained on randomly subsampled 10% (SD-T5-10) and 1% (SD-T5-1) of the full dataset, keeping sequence length distribution.
5. **Shallow SD-T5**: SD-T5 with half the number of encoder and decoder layers.
6. **Small SD-T5**: SD-T5 with hidden size reduced by half.

All models use an identical tokenizer based on the 25 standard amino acids plus 100 sentinel tokens (<extra_id_0>, etc.).

**Stage II: Pre-training**

All models are pre-trained with the same computational budget and hyperparameters:

- Optimizer: Adafactor with learning rate 1e-3, linear warmup over 10,000 steps, followed by inverse square root decay.
- Batch size: 256 sequences per device, with packing to maximize utilization.
- Maximum sequence length: 512 residues; shorter sequences are padded, longer ones truncated.
- Training volume: Exactly 10B tokens processed (effective tokens after masking/corruption).
- Dropout: 0.1 on attention and feed-forward layers.
- Mixed precision (bfloat16) is used where hardware supports.
- Perplexity on a held-out validation set (1% of training data) is monitored every 1,000 steps; training stops if no improvement for 20,000 steps.

Each variant is trained on a cluster of 32 TPU v3 cores to ensure reproducibility.

**Stage III: Downstream Evaluation**

For each pre-trained model, we extract fixed-dimensional per-residue embeddings for sequences in standard benchmarks:

- Secondary structure (3-state, CB513): For encoder-decoder models, we concatenate the encoder output and the decoder’s last hidden state after processing the uncorrupted input (i.e., without sentinel tokens). For encoder-only models, we use the final hidden states; for decoder-only, we use the output of the top layer.
- Subcellular localization (DeepLoc set): We obtain a sequence-level embedding by averaging the per-residue embeddings.

We train a single linear layer (logistic regression) on top of these frozen embeddings using the Scikit-learn default parameters. Performance is reported as Q3 accuracy (secondary structure) and multi-class classification accuracy (subcellular localization).

**Stage IV: Structural Attention Precision (SAP) Metric**

To directly quantify the structural (and co-evolutionary) information captured by each model, we propose the SAP metric, which leverages the model’s self-attention patterns.

For a protein of length L with known experimental structure (from CASP FM/CAMEO targets, or a filtered ProteinNet test set), we perform a single forward pass with the model (no corruption/masking) and extract the L×L attention matrices from the last transformer layer. For multi-head attention, we average across all heads to obtain a single attention matrix A.

We then compute precision and recall of contact prediction by thresholding A:

- True positive (TP): A_ij > threshold and Cβ distance < 8 Å in the PDB structure.
- False positive (FP): A_ij > threshold but distance ≥ 8 Å.
- False negative (FN): A_ij ≤ threshold and distance < 8 Å.

We vary the threshold from the minimum to the maximum attention value and record precision = TP/(TP+FP) vs. recall = TP/(TP+FN). The area under the precision-recall curve (AUC-PR) is the SAP score for that protein. The final model-level SAP is the mean over all evaluation proteins.

*Validation of SAP as a co-evolutionary proxy:* On a subset of proteins with sufficient homologs to construct a high-quality MSA, we compute DCA (Direct Coupling Analysis) contact scores using the plmDCA method. We then compute the Spearman correlation between the flattened upper triangle of the plmDCA DI matrix and the flattened attention matrix (only for pairs with sequence separation ≥ 6 to avoid local contacts). A strong positive correlation confirms that attention indeed captures evolutionary constraints akin to a “soft MSA”. We report this correlation alongside SAP.

**Implementation Details**

- All models are implemented using the Hugging Face Transformers library, with custom pre-training scripts.
- Data preprocessing uses the ProteinDataset class from the ESM repository.
- SAP computation is vectorized; a single evaluation run on 200 proteins takes ~1 GPU-hour after embedding extraction.
- The entire pipeline is designed to be run with a single command-line entry point, and all hyperparameters are exposed via YAML configuration files for full reproducibility.

By combining systematic ablation of the T5 design space with the SAP probing metric, the method provides a rigorous, causal explanation for the breakthrough performance of ProtT5 in single-sequence structure prediction.

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
