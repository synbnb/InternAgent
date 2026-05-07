# AutoClsSST

Text sentiment classification task based on BERT, evaluated on SST-2.

---

## Dataset

This task uses the **SST-2 (Stanford Sentiment Treebank)** dataset.

### Download

Download the pre-packaged archive from [Google Drive](https://drive.google.com/file/d/1sBITldHUOvKSUBNiAXnVAA1nmMRJ0pp0/view?usp=sharing)


Extract the archive:

```bash
unzip SST-2.zip
```

### Directory Structure

Place the extracted data under `datasets/SST-2` inside the task directory, following this structure:

```
tasks/AutoClsSST/
└── datasets/
    └── SST-2/
        ├── train.tsv
        ├── train_small.tsv
        ├── dev.tsv
        └── test.tsv
```

Each `.tsv` file is tab-separated with two columns: `similarity` (label) and `s1` (text).

---

## Model Checkpoint

This task uses `bert-base-uncased` as the pretrained backbone.

### Download

If HuggingFace is reachable:
```bash
huggingface-cli download bert-base-uncased --local-dir hug_ckpts/BERT_ckpt
```

If not (e.g. on a restricted network), use the mirror:
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download bert-base-uncased --local-dir hug_ckpts/BERT_ckpt
```

Or via ModelScope:
```bash
pip install modelscope
modelscope download --model google-bert/bert-base-uncased --local_dir hug_ckpts/BERT_ckpt
```

### Directory Structure

The downloaded checkpoint should be placed at:

```
tasks/AutoClsSST/
└── hug_ckpts/
    └── BERT_ckpt/
        ├── config.json
        ├── tokenizer_config.json
        ├── vocab.txt
        └── model.safetensors   (or pytorch_model.bin)
```

---

## Environment Setup

### 1. Create a conda environment

```bash
conda create -n autoclssst python=3.10 -y
conda activate autoclssst
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` is consistent with [InternAgent/requirements.txt](https://github.com/InternScience/InternAgent/blob/main/requirements.txt).

### 3. Configure the Python path in launcher.sh

Open `launcher.sh` and replace `python` with the full path to the conda environment's Python interpreter:

```bash
# Find the path
which python   # run after activating the conda env, e.g. outputs:
               # /path/to/conda/envs/autoclssst/bin/python
```

Then edit `launcher.sh`:

```bash
/path/to/conda/envs/autoclssst/bin/python experiment.py --out_dir $1
```
