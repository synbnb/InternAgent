# AutoMem

Improve the agentic memory system for LLM agents from [A-MEM](https://github.com/WujiangXu/A-mem), evaluated on the LoCoMo conversational QA benchmark.

---

## Dataset

The LoCoMo evaluation set (`locomo10.json`) is already bundled under `code/data/` — no download required.

---

## Environment Setup

### 1. Create a conda environment

```bash
conda create -n automem python=3.9 -y
conda activate automem
```

### 2. Install dependencies

```bash
pip install -r code/requirements.txt
```

---

## LLM Backend

The evaluation script supports two backends: **sglang** (default) and **OpenAI-compatible APIs**.

### Option A — OpenAI-compatible API

Set your API key in `.env` (at the repo root or task directory):

```
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=your-endpoint   # omit if using OpenAI directly
```

Then run with `--backend openai` and your chosen model:

```bash
python code/eval.py --backend openai --model gpt-4o-mini --out_dir $1
```

### Option B — sglang (local inference)

Start an sglang server with your model before running:

```bash
python -m sglang.launch_server --model Qwen2.5-3B-Instruct --port 30000
```

Then run the default launcher:

```bash
bash launcher.sh <out_dir>
```

The `all-MiniLM-L6-v2` sentence embedding model used for retrieval evaluation is downloaded automatically by `sentence-transformers` on first run.

---

*For full details on the A-MEM system and paper, see the [original repository](https://github.com/WujiangXu/A-mem).*
