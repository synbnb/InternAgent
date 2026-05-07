# AutoTTRL

Improve the [TTRL (Test-Time Reinforcement Learning)](https://github.com/PRIME-RL/TTRL) framework for adapting LLMs to reasoning tasks using only unlabeled test data, evaluated on AIME 2024.

---

## Dataset

The AIME-TTT dataset is bundled under `code/verl/data/AIME-TTT/` — no download required.

---

## Model Checkpoint

TTRL uses `Qwen2.5-Math-7B` as the base model. Download it before running:

```bash
huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir hug_ckpts/Qwen2.5-Math-7B
```

Or via mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir hug_ckpts/Qwen2.5-Math-7B
```

---

## Environment Setup

```bash
conda create -n ttrl python=3.10 -y
conda activate ttrl
cd code/verl
bash scripts/install_ttrl_deps.sh
pip install -e .
```

---

## Running the Experiment

The launcher references `code/verl/examples/ttrl/Qwen2.5/aime.sh`. This script is not included in the bundled code tree — clone the full TTRL repository to obtain it:

```bash
git clone https://github.com/PRIME-RL/TTRL.git
# copy or symlink TTRL/verl/examples/ into code/verl/examples/
```

Then run:

```bash
conda activate ttrl
bash code/verl/examples/ttrl/Qwen2.5/aime.sh
```

> **Compute requirement**: the original TTRL experiments were conducted on **8 × A100 80 GB GPUs**. This task is not runnable on consumer hardware.

---

*For full details on TTRL and the paper, see the [original repository](https://github.com/PRIME-RL/TTRL).*
