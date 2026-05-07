# AutoTTS

Improve the [Atom of Thoughts (AoT)](https://github.com/qixucen/atom) test-time scaling framework for LLM reasoning, evaluated on LongBench.

---

## Dataset

All evaluation datasets (`longbench`, `math`, `gsm8k`, `bbh`, `mmlu`, `hotpotqa`) are bundled under `code/experiment/data/` — no download required.

---

## API Configuration

AoT calls an OpenAI-compatible API. Create `code/apikey.py` with your credentials:

```python
url = "https://api.openai.com/v1"  # replace with your endpoint if needed
api_key = [
    "your-api-key-here",  # add multiple keys to improve concurrency
]
```

---

## Environment Setup

```bash
conda create -n autotts python=3.10 -y
conda activate autotts
pip install -r code/requirements.txt
```

---

## Running the Experiment

```bash
python code/main.py \
  --dataset longbench \
  --start 0 --end 400 \
  --model gpt-4o-mini-2024-07-18 \
  --mode atom
```

Other supported datasets: `math`, `gsm8k`, `bbh`, `mmlu`, `hotpotqa`.

---

*For full details on AoT and the paper, see the [original repository](https://github.com/qixucen/atom).*
