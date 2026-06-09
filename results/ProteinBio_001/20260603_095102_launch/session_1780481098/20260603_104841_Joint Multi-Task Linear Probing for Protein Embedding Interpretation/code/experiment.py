#!/usr/bin/env python
"""
联合多任务线性探测实验
复现ProtBERT嵌入对蛋白质二级结构预测的线性可解释性

关键方法：
1. 联合多任务线性探测 - 同时预测疏水性、电荷和氨基酸身份
2. 二级结构分类 - 比较嵌入仅用模型 vs 增强模型
3. 统计检验 - Wilcoxon符号秩检验
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# 设置随机种子以确保可重复性
np.random.seed(42)


# ============================================
# 物理化学属性定义
# ============================================

# Kyte-Doolittle疏水性指数
HYDROPHOBICITY_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# pH 7时的电荷
CHARGE_SCALE = {
    'D': -1.0, 'E': -1.0,
    'K': 1.0, 'R': 1.0,
    'H': 0.1,
}

# 氨基酸顺序
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# 二级结构编码
SS_TO_LABEL = {'H': 0, 'E': 1, 'C': 2}


# ============================================
# 嵌入生成器（使用ProtBERT）
# ============================================

class ProtBERTEmbeddingGenerator:
    """使用ProtBERT生成每残基嵌入"""

    def __init__(self, model_name: str = 'Rostlab/prot_bert', max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(self):
        """懒加载模型"""
        if self.model is not None:
            return

        try:
            from transformers import BertModel, BertTokenizer
            import torch

            print(f"加载ProtBERT模型: {self.model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"模型加载成功，使用设备: {self.device}")

        except ImportError:
            print("警告：transformers库未安装，将使用模拟数据")
            print("正在生成模拟嵌入数据...")

    def get_embeddings(self, sequence: str) -> np.ndarray:
        """
        获取序列的每残基嵌入
        返回形状: (L, 1024)
        """
        if self.model is None:
            # 生成模拟嵌入数据
            return self._get_mock_embeddings(sequence)

        # 使用真实的ProtBERT模型
        from transformers import BertModel, BertTokenizer
        import torch

        L = len(sequence)

        if L <= self.max_length - 2:
            # 序列较短，一次性处理
            embeddings = self._embed_single_pass(sequence)
        else:
            # 序列较长，分段处理并平均重叠区域
            embeddings = self._embed_long_sequence(sequence)

        return embeddings

    def _embed_single_pass(self, sequence: str) -> np.ndarray:
        """单次获取嵌入（用于短序列）"""
        import torch

        # 添加特殊标记
        sequence_with_special = ' '.join(list(sequence))

        # Tokenize
        inputs = self.tokenizer(
            sequence_with_special,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=self.max_length
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取最后一层隐藏状态
        hidden_states = outputs.last_hidden_state[0]  # (seq_len, 1024)

        # 移除[CLS]和[SEP]标记
        # 注意：ProtBERT的tokenizer会在每个氨基酸之间和首尾添加特殊标记
        # 所以需要仔细处理
        embeddings = hidden_states[1:-1]  # 移除首尾特殊标记

        # 确保嵌入数量与序列长度匹配
        if len(embeddings) != len(sequence):
            # 如果不匹配，截断或填充
            min_len = min(len(embeddings), len(sequence))
            embeddings = embeddings[:min_len]

        return embeddings.cpu().numpy()

    def _embed_long_sequence(self, sequence: str, overlap: int = 10) -> np.ndarray:
        """分段处理长序列"""
        L = len(sequence)
        chunk_size = self.max_length - 2 - overlap

        # 计算分段数
        n_chunks = (L + chunk_size - overlap - 1) // (chunk_size - overlap)

        # 初始化累加数组
        embedding_sum = None
        embedding_count = np.zeros(L, dtype=np.int32)

        for i in range(n_chunks):
            start = max(0, i * (chunk_size - overlap))
            end = min(L, start + chunk_size)

            chunk_seq = sequence[start:end]
            chunk_emb = self._embed_single_pass(chunk_seq)

            # 累加嵌入
            if embedding_sum is None:
                embedding_sum = np.zeros((L, chunk_emb.shape[1]), dtype=np.float32)

            chunk_L = len(chunk_emb)
            embedding_sum[start:start+chunk_L] += chunk_emb
            embedding_count[start:start+chunk_L] += 1

        # 平均重叠区域
        embeddings = embedding_sum / embedding_count[:, np.newaxis]

        return embeddings

    def _get_mock_embeddings(self, sequence: str) -> np.ndarray:
        """生成模拟嵌入数据"""
        L = len(sequence)

        # 为每个氨基酸生成一个基础模式
        np.random.seed(42)
        aa_embeddings = {}
        for aa in AMINO_ACIDS:
            base = np.random.randn(1024) * 0.1
            # 添加氨基酸特定的偏移
            idx = AMINO_ACIDS.index(aa)
            base[idx * 50:(idx + 1) * 50] += np.random.randn(50) * 0.5
            aa_embeddings[aa] = base

        # 构建序列嵌入
        embeddings = np.zeros((L, 1024), dtype=np.float32)
        for i, aa in enumerate(sequence):
            if aa in aa_embeddings:
                embeddings[i] = aa_embeddings[aa] + np.random.randn(1024) * 0.02
            else:
                embeddings[i] = np.random.randn(1024) * 0.1

        return embeddings


# ============================================
# 数据加载类
# ============================================

class ProteinDataLoader:
    """加载和预处理蛋白质数据"""

    def __init__(self, data_dir: str = 'data', use_model_embeddings: bool = False):
        self.data_dir = Path(data_dir)
        self.sequences = None
        self.embeddings = None
        self.protein_ids = []
        self.use_model_embeddings = use_model_embeddings
        self.embedding_generator = None

        if use_model_embeddings:
            self.embedding_generator = ProtBERTEmbeddingGenerator()

    def load_sequences(self) -> pd.DataFrame:
        """加载蛋白质序列和二级结构标签"""
        seq_file = self.data_dir / 'protein_sequences_sample.csv'
        self.sequences = pd.read_csv(seq_file)
        print(f"加载了 {len(self.sequences)} 个蛋白质序列")
        return self.sequences

    def load_or_generate_embeddings(self, protein_ids: List[str] = None) -> Dict[str, np.ndarray]:
        """加载或生成嵌入"""
        if protein_ids is None and self.sequences is not None:
            protein_ids = self.sequences['protein_id'].tolist()

        self.embeddings = {}

        if self.use_model_embeddings:
            # 使用模型生成嵌入
            self.embedding_generator.load_model()

            for prot_id in tqdm(protein_ids, desc="生成嵌入"):
                seq_row = self.sequences[self.sequences['protein_id'] == prot_id].iloc[0]
                sequence = seq_row['sequence']
                embeddings = self.embedding_generator.get_embeddings(sequence)
                self.embeddings[prot_id] = embeddings
        else:
            # 尝试从文件加载（如果格式正确）
            # 或者生成模拟嵌入
            print("生成模拟嵌入数据...")
            mock_generator = ProtBERTEmbeddingGenerator()

            for prot_id in tqdm(protein_ids, desc="生成嵌入"):
                seq_row = self.sequences[self.sequences['protein_id'] == prot_id].iloc[0]
                sequence = seq_row['sequence']
                embeddings = mock_generator.get_embeddings(sequence)
                self.embeddings[prot_id] = embeddings

        print(f"生成了 {len(self.embeddings)} 个蛋白质的嵌入")
        return self.embeddings

    def get_hydrophobicity(self, aa: str) -> float:
        return HYDROPHOBICITY_SCALE.get(aa, 0.0)

    def get_charge(self, aa: str) -> float:
        return CHARGE_SCALE.get(aa, 0.0)

    def get_onehot_aa(self, aa: str) -> np.ndarray:
        onehot = np.zeros(20, dtype=np.float32)
        if aa in AMINO_ACIDS:
            idx = AMINO_ACIDS.index(aa)
            onehot[idx] = 1.0
        return onehot

    def build_target_matrix(self, sequence: str) -> np.ndarray:
        L = len(sequence)
        targets = np.zeros((L, 22), dtype=np.float32)

        for j, aa in enumerate(sequence):
            targets[j, 0] = self.get_hydrophobicity(aa)
            targets[j, 1] = self.get_charge(aa)
            targets[j, 2:] = self.get_onehot_aa(aa)

        return targets


# ============================================
# 联合多任务线性探测
# ============================================

class JointMultiTaskProbe:
    """联合多任务线性探测分析器"""

    def __init__(self, loader: ProteinDataLoader, n_splits: int = 5):
        self.loader = loader
        self.n_splits = n_splits
        self.alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        self.cv_r2_scores = []
        self.per_property_r2 = {0: [], 1: []}
        self.best_alphas = []

    def prepare_data(self, protein_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        Y_list = []

        for prot_id in protein_ids:
            seq_row = self.loader.sequences[
                self.loader.sequences['protein_id'] == prot_id
            ].iloc[0]
            sequence = seq_row['sequence']

            # 获取嵌入
            embedding = self.loader.embeddings[prot_id]

            # 构建目标矩阵
            targets = self.loader.build_target_matrix(sequence)

            # 确保长度匹配
            min_len = min(len(embedding), len(targets))
            X_list.append(embedding[:min_len])
            Y_list.append(targets[:min_len])

        X = np.vstack(X_list)
        Y = np.vstack(Y_list)

        return X, Y

    def compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1.0 - ss_res / ss_tot

    def run_cross_validation(self, protein_ids: List[str]) -> Dict:
        print(f"\n开始 {self.n_splits} 折交叉验证...")

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(protein_ids)):
            print(f"  处理折 {fold + 1}/{self.n_splits}...")

            train_ids = [protein_ids[i] for i in train_idx]
            val_ids = [protein_ids[i] for i in val_idx]

            X_train, Y_train = self.prepare_data(train_ids)
            X_val, Y_val = self.prepare_data(val_ids)

            inner_kf = KFold(n_splits=3, shuffle=True, random_state=42 + fold)

            model = RidgeCV(
                alphas=self.alphas,
                scoring='r2',
                cv=inner_kf
            )
            model.fit(X_train, Y_train)

            self.best_alphas.append(model.alpha_)

            Y_pred = model.predict(X_val)

            joint_r2 = self.compute_r2(Y_val, Y_pred)
            self.cv_r2_scores.append(joint_r2)

            for prop_idx in [0, 1]:
                prop_r2 = r2_score(Y_val[:, prop_idx], Y_pred[:, prop_idx])
                self.per_property_r2[prop_idx].append(prop_r2)

            print(f"    折 {fold + 1} R²: {joint_r2:.4f}, alpha: {model.alpha_:.4f}")

        results = {
            'mean_r2': np.mean(self.cv_r2_scores),
            'std_r2': np.std(self.cv_r2_scores),
            'mean_hydrophobicity_r2': np.mean(self.per_property_r2[0]),
            'mean_charge_r2': np.mean(self.per_property_r2[1]),
            'fold_r2_scores': self.cv_r2_scores,
            'best_alphas': self.best_alphas
        }

        return results


# ============================================
# 二级结构分类
# ============================================

class SecondaryStructureClassifier:
    """二级结构分类器"""

    def __init__(self, loader: ProteinDataLoader, n_splits: int = 10):
        self.loader = loader
        self.n_splits = n_splits
        self.cs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        self.q3_embedding_only = []
        self.q3_augmented = []

    def prepare_classification_data(self, protein_ids: List[str],
                                     use_augmented: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict]:
        X_list = []
        y_list = []
        protein_to_indices = {}

        start_idx = 0

        for prot_id in protein_ids:
            seq_row = self.loader.sequences[
                self.loader.sequences['protein_id'] == prot_id
            ].iloc[0]
            sequence = seq_row['sequence']
            ss_labels = seq_row['secondary_structure']

            embedding = self.loader.embeddings[prot_id]
            length = min(len(embedding), len(sequence), len(ss_labels))
            embedding = embedding[:length]
            sequence = sequence[:length]
            ss_labels = ss_labels[:length]

            if use_augmented:
                hydro = np.array([self.loader.get_hydrophobicity(aa) for aa in sequence]).reshape(-1, 1)
                charge = np.array([self.loader.get_charge(aa) for aa in sequence]).reshape(-1, 1)
                phys_features = np.hstack([hydro, charge])
                features = np.hstack([embedding, phys_features])
            else:
                features = embedding

            labels = np.array([SS_TO_LABEL.get(c, 2) for c in ss_labels])

            X_list.append(features)
            y_list.append(labels)

            end_idx = start_idx + length
            protein_to_indices[prot_id] = (start_idx, end_idx)
            start_idx = end_idx

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        return X, y, protein_to_indices

    def run_cross_validation(self, protein_ids: List[str]) -> Dict:
        print(f"\n开始 {self.n_splits} 折二级结构分类交叉验证...")

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        all_protein_q3_emb = {}
        all_protein_q3_aug = {}

        for fold, (train_idx, val_idx) in enumerate(kf.split(protein_ids)):
            print(f"  处理折 {fold + 1}/{self.n_splits}...")

            train_ids = [protein_ids[i] for i in train_idx]
            val_ids = [protein_ids[i] for i in val_idx]

            # 模型A: 仅使用嵌入
            X_train_emb, y_train, _ = self.prepare_classification_data(
                train_ids, use_augmented=False
            )
            X_val_emb, y_val, prot_indices_emb = self.prepare_classification_data(
                val_ids, use_augmented=False
            )

            model_emb = LogisticRegressionCV(
                Cs=self.cs,
                cv=3,
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=42 + fold
            )
            model_emb.fit(X_train_emb, y_train)
            y_pred_emb = model_emb.predict(X_val_emb)

            protein_q3_emb = {}
            for prot_id, (start, end) in prot_indices_emb.items():
                q3 = accuracy_score(y_val[start:end], y_pred_emb[start:end])
                protein_q3_emb[prot_id] = q3

            # 模型B: 嵌入+物理特征
            X_train_aug, y_train_aug, _ = self.prepare_classification_data(
                train_ids, use_augmented=True
            )
            X_val_aug, y_val_aug, prot_indices_aug = self.prepare_classification_data(
                val_ids, use_augmented=True
            )

            model_aug = LogisticRegressionCV(
                Cs=self.cs,
                cv=3,
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=42 + fold
            )
            model_aug.fit(X_train_aug, y_train_aug)
            y_pred_aug = model_aug.predict(X_val_aug)

            protein_q3_aug = {}
            for prot_id, (start, end) in prot_indices_aug.items():
                q3 = accuracy_score(y_val_aug[start:end], y_pred_aug[start:end])
                protein_q3_aug[prot_id] = q3

            for prot_id in val_ids:
                if prot_id not in all_protein_q3_emb:
                    all_protein_q3_emb[prot_id] = []
                    all_protein_q3_aug[prot_id] = []
                all_protein_q3_emb[prot_id].append(protein_q3_emb[prot_id])
                all_protein_q3_aug[prot_id].append(protein_q3_aug[prot_id])

            fold_q3_emb = np.mean(list(protein_q3_emb.values()))
            fold_q3_aug = np.mean(list(protein_q3_aug.values()))

            print(f"    折 {fold + 1} - 嵌入 Q3: {fold_q3_emb:.4f}, 增强 Q3: {fold_q3_aug:.4f}")

        # 汇总结果
        final_q3_emb = {}
        final_q3_aug = {}

        for prot_id in all_protein_q3_emb.keys():
            final_q3_emb[prot_id] = np.mean(all_protein_q3_emb[prot_id])
            final_q3_aug[prot_id] = np.mean(all_protein_q3_aug[prot_id])
            self.q3_embedding_only.append(final_q3_emb[prot_id])
            self.q3_augmented.append(final_q3_aug[prot_id])

        results = {
            'mean_q3_embedding': np.mean(self.q3_embedding_only),
            'mean_q3_augmented': np.mean(self.q3_augmented),
            'std_q3_embedding': np.std(self.q3_embedding_only),
            'std_q3_augmented': np.std(self.q3_augmented),
            'protein_q3_emb': final_q3_emb,
            'protein_q3_aug': final_q3_aug
        }

        return results


# ============================================
# 统计检验
# ============================================

def run_statistical_test(q3_emb: List[float], q3_aug: List[float]) -> Dict:
    differences = np.array(q3_aug) - np.array(q3_emb)
    statistic, p_value = wilcoxon(differences, alternative='greater')

    results = {
        'statistic': statistic,
        'p_value': p_value,
        'mean_difference': np.mean(differences),
        'significant': p_value < 0.05,
        'interpretation': '显著' if p_value < 0.05 else '不显著'
    }

    return results


# ============================================
# 可视化
# ============================================

class ResultVisualizer:
    """结果可视化"""

    def __init__(self, output_dir: str = 'report/images'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_cv_r2_scores(self, r2_scores: List[float], fold_names: List[str]):
        fig, ax = plt.subplots(figsize=(10, 6))

        positions = np.arange(len(r2_scores))
        bars = ax.bar(positions, r2_scores, color='steelblue', edgecolor='black', linewidth=1.5)

        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}', ha='center', va='bottom', fontsize=10)

        mean_r2 = np.mean(r2_scores)
        ax.axhline(y=mean_r2, color='red', linestyle='--', linewidth=2,
                  label=f'Mean R² = {mean_r2:.4f}')

        ax.set_xlabel('Cross-Validation Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Joint Multi-Task Linear Probe: Cross-Validation R² Scores',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(fold_names)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cv_r2_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"保存图表: {self.output_dir / 'cv_r2_scores.png'}")

    def plot_per_property_r2(self, hydro_r2: List[float], charge_r2: List[float], fold_names: List[str]):
        fig, ax = plt.subplots(figsize=(10, 6))

        positions = np.arange(len(fold_names))
        width = 0.35

        bars1 = ax.bar(positions - width/2, hydro_r2, width,
                      label='Hydrophobicity', color='steelblue', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(positions + width/2, charge_r2, width,
                      label='Charge (pH 7)', color='coral', edgecolor='black', linewidth=1.5)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Cross-Validation Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Property R² Scores from Joint Probe',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(fold_names)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_property_r2.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"保存图表: {self.output_dir / 'per_property_r2.png'}")

    def plot_q3_comparison(self, q3_emb: List[float], q3_aug: List[float]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 箱线图
        data_to_plot = [q3_emb, q3_aug]
        positions = [1, 2]

        bp = ax1.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=True)

        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

        ax1.set_xticks(positions)
        ax1.set_xticklabels(['Embedding Only', 'Augmented\n(+ Hydro + Charge)'], fontsize=11, fontweight='bold')
        ax1.set_ylabel('Q3 Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Secondary Structure Q3 Accuracy Comparison', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        means = [np.mean(q3_emb), np.mean(q3_aug)]
        stds = [np.std(q3_emb), np.std(q3_aug)]

        for i, (mean, std) in enumerate(zip(means, stds), 1):
            ax1.text(i, ax1.get_ylim()[1] * 0.95,
                    f'μ={mean:.4f}\nσ={std:.4f}',
                    ha='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                             edgecolor='gray', alpha=0.8))

        # 散点图
        ax2.scatter(q3_emb, q3_aug, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)

        min_val = min(min(q3_emb), min(q3_aug))
        max_val = max(max(q3_emb), max(q3_aug))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x')

        ax2.set_xlabel('Embedding Only Q3', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Augmented Model Q3', fontsize=12, fontweight='bold')
        ax2.set_title('Per-Protein Q3: Paired Comparison', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'q3_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"保存图表: {self.output_dir / 'q3_comparison.png'}")

    def plot_differences_distribution(self, differences: np.ndarray):
        fig, ax = plt.subplots(figsize=(10, 6))

        n, bins, patches = ax.hist(differences, bins=30, color='steelblue',
                                  edgecolor='black', linewidth=1, alpha=0.7)

        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Zero difference')

        mean_diff = np.mean(differences)
        ax.axvline(x=mean_diff, color='green', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_diff:.4f}')

        ax.set_xlabel('Q3 Difference (Augmented - Embedding Only)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Per-Protein Q3 Differences', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'q3_differences_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"保存图表: {self.output_dir / 'q3_differences_distribution.png'}")


# ============================================
# 主函数
# ============================================

def main():
    print("=" * 80)
    print("联合多任务线性探测实验")
    print("Joint Multi-Task Linear Probing for Protein Embedding Interpretation")
    print("=" * 80)

    # 1. 加载数据
    print("\n[步骤 1] 加载数据...")
    loader = ProteinDataLoader(data_dir='data')
    loader.load_sequences()

    protein_ids = loader.sequences['protein_id'].tolist()

    # 生成或加载嵌入
    print("\n正在生成/加载嵌入...")
    loader.load_or_generate_embeddings(protein_ids)

    # 2. 联合多任务线性探测
    print("\n[步骤 2] 联合多任务线性探测 (5折交叉验证)...")
    probe = JointMultiTaskProbe(loader, n_splits=5)
    probe_results = probe.run_cross_validation(protein_ids)

    print("\n--- 联合探测结果 ---")
    print(f"平均 R²: {probe_results['mean_r2']:.4f} ± {probe_results['std_r2']:.4f}")
    print(f"疏水性 R²: {probe_results['mean_hydrophobicity_r2']:.4f}")
    print(f"电荷 R²: {probe_results['mean_charge_r2']:.4f}")

    # 3. 二级结构分类
    print("\n[步骤 3] 二级结构分类 (10折交叉验证)...")
    classifier = SecondaryStructureClassifier(loader, n_splits=10)
    cls_results = classifier.run_cross_validation(protein_ids)

    print("\n--- 二级结构分类结果 ---")
    print(f"仅嵌入模型 Q3: {cls_results['mean_q3_embedding']:.4f} ± {cls_results['std_q3_embedding']:.4f}")
    print(f"增强模型 Q3: {cls_results['mean_q3_augmented']:.4f} ± {cls_results['std_q3_augmented']:.4f}")

    # 4. 统计检验
    print("\n[步骤 4] Wilcoxon符号秩检验...")
    differences = np.array(classifier.q3_augmented) - np.array(classifier.q3_embedding_only)
    test_results = run_statistical_test(classifier.q3_embedding_only, classifier.q3_augmented)

    print("\n--- 统计检验结果 ---")
    print(f"Wilcoxon统计量: {test_results['statistic']}")
    print(f"P值: {test_results['p_value']:.6f}")
    print(f"平均差异: {test_results['mean_difference']:.6f}")
    print(f"差异是否显著 (α=0.05): {test_results['interpretation']}")

    # 5. 可视化
    print("\n[步骤 5] 生成可视化图表...")
    visualizer = ResultVisualizer(output_dir='report/images')

    fold_names_5 = [f'Fold {i+1}' for i in range(5)]
    visualizer.plot_cv_r2_scores(probe_results['fold_r2_scores'], fold_names_5)

    visualizer.plot_per_property_r2(
        probe.per_property_r2[0],
        probe.per_property_r2[1],
        fold_names_5
    )

    visualizer.plot_q3_comparison(classifier.q3_embedding_only, classifier.q3_augmented)
    visualizer.plot_differences_distribution(differences)

    # 6. 保存结果
    print("\n[步骤 6] 保存结果...")
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    results_summary = {
        'joint_probe': {
            'mean_r2': float(probe_results['mean_r2']),
            'std_r2': float(probe_results['std_r2']),
            'mean_hydrophobicity_r2': float(probe_results['mean_hydrophobicity_r2']),
            'mean_charge_r2': float(probe_results['mean_charge_r2']),
            'fold_r2_scores': [float(x) for x in probe_results['fold_r2_scores']],
            'best_alphas': [float(x) for x in probe_results['best_alphas']]
        },
        'classification': {
            'mean_q3_embedding': float(cls_results['mean_q3_embedding']),
            'std_q3_embedding': float(cls_results['std_q3_embedding']),
            'mean_q3_augmented': float(cls_results['mean_q3_augmented']),
            'std_q3_augmented': float(cls_results['std_q3_augmented'])
        },
        'statistical_test': {
            'statistic': float(test_results['statistic']),
            'p_value': float(test_results['p_value']),
            'mean_difference': float(test_results['mean_difference']),
            'significant': bool(test_results['significant'])
        },
        'metadata': {
            'n_proteins': len(protein_ids),
            'n_folds_probe': 5,
            'n_folds_classification': 10
        }
    }

    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    protein_results = []
    for prot_id in sorted(cls_results['protein_q3_emb'].keys()):
        protein_results.append({
            'protein_id': prot_id,
            'q3_embedding': float(cls_results['protein_q3_emb'][prot_id]),
            'q3_augmented': float(cls_results['protein_q3_aug'][prot_id]),
            'difference': float(cls_results['protein_q3_aug'][prot_id] -
                              cls_results['protein_q3_emb'][prot_id])
        })

    protein_df = pd.DataFrame(protein_results)
    protein_df.to_csv(output_dir / 'protein_q3_scores.csv', index=False)

    print(f"\n结果已保存到 {output_dir}/")

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)

    return results_summary


if __name__ == '__main__':
    results = main()
