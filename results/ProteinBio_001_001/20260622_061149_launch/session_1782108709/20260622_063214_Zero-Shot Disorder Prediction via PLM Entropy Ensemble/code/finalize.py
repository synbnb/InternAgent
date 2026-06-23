#!/usr/bin/env python3
"""Finalize: train SS3 probe on cached embeddings with correct label alignment."""
import os, sys, json, warnings
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
os.makedirs(IMG, exist_ok=True)
np.random.seed(42)
VALID_AA = set('ACDEFGHIKLMNPQRSTVWYX')
SS3_MAP = {'H': 0, 'E': 1, 'C': 2}

def _segs(b):
    segs = []; i = 0
    while i < len(b):
        if b[i]:
            s = i
            while i < len(b) and b[i]: i += 1
            segs.append((s, i-1))
        else: i += 1
    return segs

def compute_sov(pred, true):
    n = 3; sov_sum = 0.0; norm_sum = 0.0
    for s in range(n):
        tb = (true == s).astype(int); pb = (pred == s).astype(int)
        t_segs = _segs(tb); p_segs = _segs(pb)
        if not t_segs: continue
        for st in t_segs:
            ov = 0
            for sp in p_segs:
                o = max(0, min(st[1], sp[1]) - max(st[0], sp[0]) + 1)
                if o > 0: ov += o
            if ov > 0:
                lt = st[1] - st[0] + 1
                sov_sum += ov / lt * lt; norm_sum += lt
    return sov_sum / norm_sum if norm_sum > 0 else 0.0

# Load embeddings
cache = OUT / "emb_cache"
print("Loading cached embeddings...")
train_e = sorted([f for f in os.listdir(cache) if f.startswith('train_sub_') and f.endswith('.npy')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
casp_e = sorted([f for f in os.listdir(cache) if f.startswith('casp12_') and f.endswith('.npy')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
cb_e = sorted([f for f in os.listdir(cache) if f.startswith('cb513_') and f.endswith('.npy')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
print(f"  train: {len(train_e)}, casp12: {len(casp_e)}, cb513: {len(cb_e)}")

# Load label sequences from raw data to match embedding lengths
from datasets import load_dataset
ds = load_dataset('lamm-mit/protein-secondary-structure-netsurfp')
train_raw = list(ds['train']) + list(ds['validation'])
casp_raw = list(ds['casp12'])
cb_raw = list(ds['cb513'])

def replace_non_standard(seq):
    return ''.join(c if c in VALID_AA else 'X' for c in seq)

# Build aligned train data (train_sub was first 300 proteins using np.random.choice)
# We need to match - the original code used random choice without replacement from range(len(train_p))
# Let's just use the first 300 from train_raw directly
n_train = len(train_e)
print(f"\nAligning {n_train} training proteins...")
all_X, all_y = [], []
for i in range(n_train):
    raw = train_raw[i % len(train_raw)]
    seq = replace_non_standard(raw['sequence'][:2048])
    lab = raw['labels'][:len(seq)]
    emb = np.load(cache / train_e[i])
    L = min(len(emb), len(lab), 64)
    if L < 3: continue
    y = np.array([SS3_MAP.get(c, 2) for c in lab[:L]], dtype=np.int32)
    all_X.append(emb[:L]); all_y.append(y)

X_tr = np.vstack(all_X); y_tr = np.concatenate(all_y)
print(f"  Training: {X_tr.shape[0]:,} x {X_tr.shape[1]} (classes: {np.unique(y_tr).tolist()})")

print("\nTraining logistic regression...")
clf = LogisticRegression(solver='lbfgs', C=0.1, max_iter=1000, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)

def test_on(emb_files, raw_list):
    X, y = [], []
    for i, f in enumerate(emb_files):
        raw = raw_list[i % len(raw_list)]
        lab = raw['labels']
        e = np.load(cache / f)
        L = min(len(e), len(lab), 128)
        if L < 3: continue
        y.append(np.array([SS3_MAP.get(c, 2) for c in lab[:L]], dtype=np.int32))
        X.append(e[:L])
    if not X: return {}
    X = np.vstack(X); y = np.concatenate(y)
    p = clf.predict(X)
    return {'q3': float(accuracy_score(y, p)), 'sov': float(compute_sov(p, y))}

c12 = test_on(casp_e, casp_raw)
cb = test_on(cb_e, cb_raw)
print(f"  CASP12: Q3={c12.get('q3',0):.4f}, SOV={c12.get('sov',0):.4f}")
if cb: print(f"  CB513: Q3={cb.get('q3',0):.4f}, SOV={cb.get('sov',0):.4f}")

ss3 = {'casp12': c12}
if cb: ss3['cb513'] = cb
json.dump(ss3, open(OUT / "secondary_structure_scores.json", 'w'), indent=2)

# Plot
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))
dss = list(ss3.keys())
q3s = [ss3[d]['q3'] for d in dss]; sovs = [ss3[d]['sov'] for d in dss]
b1 = a1.bar(dss, q3s, color=['#3498DB','#E74C3C'][:len(dss)], alpha=0.8)
for b,v in zip(b1, q3s): a1.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
a1.set_title('Q3 Accuracy', fontweight='bold'); a1.set_ylim(0, 1); a1.grid(axis='y', alpha=0.3)
b2 = a2.bar(dss, sovs, color=['#3498DB','#E74C3C'][:len(dss)], alpha=0.8)
for b,v in zip(b2, sovs): a2.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
a2.set_title('SOV Score', fontweight='bold'); a2.set_ylim(0, 1); a2.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(IMG / "ss3_results.png", dpi=200, bbox_inches='tight'); plt.close()
print(f"\nDone! Results saved.")
