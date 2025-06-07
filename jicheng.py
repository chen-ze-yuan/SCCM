"""
集成学习模型
包含十种基础分类器和元模型分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from typing import List, Dict, Tuple, Optional
import joblib
import os


class BaseClassifierWrapper:
    """基础分类器包装器"""

    def __init__(self, classifier, name: str):
        self.classifier = classifier
        self.name = name
        self.is_fitted = False

    def fit(self, X, y):
        """训练分类器"""
        self.classifier.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError(f"Classifier {self.name} is not fitted yet.")
        return self.classifier.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError(f"Classifier {self.name} is not fitted yet.")

        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X)
        else:
            # 对于不支持概率预测的分类器，使用决策函数
            if hasattr(self.classifier, 'decision_function'):
                scores = self.classifier.decision_function(X)
                if scores.ndim == 1:
                    # 二分类情况
                    proba = np.column_stack([1 - scores, scores])
                else:
                    # 多分类情况
                    proba = F.softmax(torch.tensor(scores), dim=1).numpy()
                return proba
            else:
                # 最后的备选方案：使用硬预测
                predictions = self.predict(X)
                n_classes = len(np.unique(predictions))
                proba = np.zeros((len(predictions), n_classes))
                for i, pred in enumerate(predictions):
                    proba[i, pred] = 1.0
                return proba

    def save(self, filepath: str):
        """保存模型"""
        joblib.dump(self.classifier, filepath)

    def load(self, filepath: str):
        """加载模型"""
        self.classifier = joblib.load(filepath)
        self.is_fitted = True


class EnsembleClassifier:
    """集成分类器"""

    def __init__(self, n_classes: int, random_state: int = 42):
        self.n_classes = n_classes
        self.random_state = random_state

        # 十种基础分类器
        self.base_classifiers = self._create_base_classifiers()

        # 元模型分类器
        self.meta_classifier = self._create_meta_classifier()

        self.is_fitted = False

    def _create_base_classifiers(self) -> List[BaseClassifierWrapper]:
        """创建十种基础分类器"""
        classifiers = []

        # 1. 随机森林
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        classifiers.append(BaseClassifierWrapper(rf, "RandomForest"))

        # 2. 梯度提升
        gb = GradientBoostingClassifier(
            n_estimators=100,
            random_state=self.random_state
        )
        classifiers.append(BaseClassifierWrapper(gb, "GradientBoosting"))

        # 3. AdaBoost
        ada = AdaBoostClassifier(
            n_estimators=100,
            random_state=self.random_state
        )
        classifiers.append(BaseClassifierWrapper(ada, "AdaBoost"))

        # 4. MLP
        mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, activation='relu')
        classifiers.append(BaseClassifierWrapper(mlp, "MLPClassifier"))

        # 5. 逻辑回归
        lr = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        classifiers.append(BaseClassifierWrapper(lr, "LogisticRegression"))

        # 6. Bagging
        bag = BaggingClassifier(n_estimators=50, max_samples=0.8)
        classifiers.append(BaseClassifierWrapper(bag, "BaggingClassifier"))

        # 7. 支持向量机
        svm = SVC(
            probability=True,
            random_state=self.random_state
        )
        classifiers.append(BaseClassifierWrapper(svm, "SVM"))

        # 8. 朴素贝叶斯
        nb = GaussianNB()
        classifiers.append(BaseClassifierWrapper(nb, "NaiveBayes"))

        # 9. K近邻
        knn = KNeighborsClassifier(n_neighbors=5)
        classifiers.append(BaseClassifierWrapper(knn, "KNN"))

        # 10. 决策树
        dt = DecisionTreeClassifier(random_state=self.random_state)
        classifiers.append(BaseClassifierWrapper(dt, "DecisionTree"))

        return classifiers

    def _create_meta_classifier(self) -> BaseClassifierWrapper:
        """创建元模型分类器"""
        # 使用随机森林作为元分类器
        rad = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=3, random_state=42)
        return BaseClassifierWrapper(rad, "MetaClassifier")

    def fit(self, X, y, validation_split: float = 0.2):
        """
        训练集成模型

        Args:
            X: (n_samples, n_features) 训练特征
            y: (n_samples,) 训练标签
            validation_split: 验证集比例，用于训练元分类器
        """
        X = np.array(X)
        y = np.array(y)

        # 分割训练集和验证集
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        print("Training base classifiers...")

        # 训练基础分类器
        for i, classifier in enumerate(self.base_classifiers):
            print(f"Training {classifier.name}...")
            classifier.fit(X_train, y_train)

            # 评估基础分类器
            val_pred = classifier.predict(X_val)
            accuracy = accuracy_score(y_val, val_pred)
            print(f"{classifier.name} validation accuracy: {accuracy:.4f}")

        # 生成元特征
        print("Generating meta features...")
        meta_features = self._generate_meta_features(X_val)

        # 训练元分类器
        print("Training meta classifier...")
        self.meta_classifier = self._create_meta_classifier()
        self.meta_classifier.fit(meta_features, y_val)

        # 评估元分类器
        meta_pred = self.meta_classifier.predict(meta_features)
        meta_accuracy = accuracy_score(y_val, meta_pred)
        print(f"Meta classifier validation accuracy: {meta_accuracy:.4f}")

        self.is_fitted = True

        return self

    def _generate_meta_features(self, X):
        """生成元特征"""
        meta_features = []

        for classifier in self.base_classifiers:
            if classifier.is_fitted:
                # 使用概率预测作为元特征
                proba = classifier.predict_proba(X)
                meta_features.append(proba)

        # 拼接所有基础分类器的预测概率
        meta_features = np.concatenate(meta_features, axis=1)

        return meta_features

    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("Ensemble classifier is not fitted yet.")

        # 生成元特征
        meta_features = self._generate_meta_features(X)

        # 元分类器预测
        predictions = self.meta_classifier.predict(meta_features)

        return predictions

    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("Ensemble classifier is not fitted yet.")

        # 生成元特征
        meta_features = self._generate_meta_features(X)

        # 元分类器预测概率
        probabilities = self.meta_classifier.predict_proba(meta_features)

        return probabilities

    def get_base_predictions(self, X):
        """获取所有基础分类器的预测"""
        predictions = {}

        for classifier in self.base_classifiers:
            if classifier.is_fitted:
                pred = classifier.predict(X)
                proba = classifier.predict_proba(X)
                predictions[classifier.name] = {
                    'predictions': pred,
                    'probabilities': proba
                }

        return predictions

    def save(self, save_dir: str):
        """保存整个集成模型"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存基础分类器
        for classifier in self.base_classifiers:
            if classifier.is_fitted:
                filepath = os.path.join(save_dir, f"{classifier.name}.pkl")
                classifier.save(filepath)

        # 保存元分类器
        if self.meta_classifier and self.meta_classifier.is_fitted:
            meta_filepath = os.path.join(save_dir, "meta_classifier.pkl")
            self.meta_classifier.save(meta_filepath)

        # 保存配置
        config = {
            'n_classes': self.n_classes,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
        config_filepath = os.path.join(save_dir, "config.pkl")
        joblib.dump(config, config_filepath)

        print(f"Ensemble model saved to {save_dir}")

    def load(self, save_dir: str):
        """加载整个集成模型"""
        # 加载配置
        config_filepath = os.path.join(save_dir, "config.pkl")
        config = joblib.load(config_filepath)

        self.n_classes = config['n_classes']
        self.random_state = config['random_state']
        self.is_fitted = config['is_fitted']

        # 重新创建基础分类器
        self.base_classifiers = self._create_base_classifiers()

        # 加载基础分类器
        for classifier in self.base_classifiers:
            filepath = os.path.join(save_dir, f"{classifier.name}.pkl")
            if os.path.exists(filepath):
                classifier.load(filepath)

        # 加载元分类器
        meta_filepath = os.path.join(save_dir, "meta_classifier.pkl")
        if os.path.exists(meta_filepath):
            self.meta_classifier = self._create_meta_classifier()
            self.meta_classifier.load(meta_filepath)

        print(f"Ensemble model loaded from {save_dir}")


class PinyinEnsembleModel:
    """拼音集成模型"""

    def __init__(self, pinyin_vocab_size: int, feature_dim: int = 4):
        """
        Args:
            pinyin_vocab_size: 拼音词汇表大小
            feature_dim: 输入特征维度（预测拼音、声韵母拼音、文本拼音、真实拼音）
        """
        self.pinyin_vocab_size = pinyin_vocab_size
        self.feature_dim = feature_dim

        # 集成分类器
        self.ensemble = EnsembleClassifier(n_classes=pinyin_vocab_size)

    def prepare_features(self,
                         predicted_pinyin: List[List[int]],
                         shengyun_pinyin: List[List[int]],
                         text_pinyin: List[List[int]],
                         true_pinyin: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练特征

        Args:
            predicted_pinyin: 模型预测的拼音序列
            shengyun_pinyin: 声韵母转换的拼音序列
            text_pinyin: 文本转换的拼音序列
            true_pinyin: 真实拼音序列

        Returns:
            X: (n_samples, feature_dim) 特征矩阵
            y: (n_samples,) 标签向量
        """
        X = []
        y = []

        # 确保所有序列长度相同
        max_len = max(
            len(predicted_pinyin), len(shengyun_pinyin),
            len(text_pinyin), len(true_pinyin)
        )
        pad = 0
        for i in range(max_len):

            # 对齐序列长度
            pred_seq = predicted_pinyin[i] if i < len(predicted_pinyin) else [pad]
            shengyun_seq = shengyun_pinyin[i] if i < len(shengyun_pinyin) else [pad]
            text_seq = text_pinyin[i] if i < len(text_pinyin) else [pad]
            true_seq = true_pinyin[i] if i < len(true_pinyin) else [pad]
            seq_len = max(len(pred_seq), len(shengyun_seq), len(text_seq), len(true_seq))

            for j in range(seq_len):
                # 特征：[预测拼音, 声韵母拼音, 文本拼音, 真实拼音]
                features = [
                    pred_seq[j],
                    shengyun_seq[j] if j < len(shengyun_seq) else 0,
                    text_seq[j] if j < len(text_seq) else 0
                ]
                label = true_seq[j] if j < len(true_seq) else pad

                X.append(features)
                y.append(true_seq[j])  # 目标是真实拼音

        return np.array(X), np.array(y)

    def fit(self,
            predicted_pinyin: List[List[int]],
            shengyun_pinyin: List[List[int]],
            text_pinyin: List[List[int]],
            true_pinyin: List[List[int]]):
        """训练集成模型"""

        # 准备特征
        X, y = self.prepare_features(predicted_pinyin, shengyun_pinyin, text_pinyin, true_pinyin)

        print(f"Training ensemble model with {len(X)} samples...")

        # 训练集成分类器
        self.ensemble.fit(X, y)

        return self

    def predict(self,
                predicted_pinyin: List[List[int]],
                shengyun_pinyin: List[List[int]],
                text_pinyin: List[List[int]]) -> List[List[int]]:
        """预测最终拼音序列"""

        # 准备特征（没有真实拼音，用0填充）
        X = []
        sequence_info = []  # 记录序列信息用于重构

        for i in range(len(predicted_pinyin)):
            pred_seq = predicted_pinyin[i]
            shengyun_seq = shengyun_pinyin[i] if i < len(shengyun_pinyin) else []
            text_seq = text_pinyin[i] if i < len(text_pinyin) else []

            seq_len = max(len(pred_seq), len(shengyun_seq), len(text_seq))
            sequence_info.append((i, seq_len))

            for j in range(seq_len):
                features = [
                    pred_seq[j] if j < len(pred_seq) else 0,
                    shengyun_seq[j] if j < len(shengyun_seq) else 0,
                    text_seq[j] if j < len(text_seq) else 0,
                    0  # 占位符，预测时不知道真实拼音
                ]
                X.append(features)

        if len(X) == 0:
            return []

        # 预测
        X = np.array(X)
        predictions = self.ensemble.predict(X)

        # 重构序列
        result = []
        idx = 0
        for seq_idx, seq_len in sequence_info:
            seq_predictions = predictions[idx:idx + seq_len].tolist()
            result.append(seq_predictions)
            idx += seq_len

        return result

    def evaluate(self,
                 predicted_pinyin: List[List[int]],
                 shengyun_pinyin: List[List[int]],
                 text_pinyin: List[List[int]],
                 true_pinyin: List[List[int]]) -> Dict[str, float]:
        """评估集成模型性能"""

        # 准备特征
        X, y = self.prepare_features(predicted_pinyin, shengyun_pinyin, text_pinyin, true_pinyin)

        if len(X) == 0:
            return {'accuracy': 0.0}

        # 预测
        predictions = self.ensemble.predict(X)

        # 计算准确率
        accuracy = accuracy_score(y, predictions)

        # 获取基础分类器预测
        base_predictions = self.ensemble.get_base_predictions(X)
        base_accuracies = {}

        for name, pred_info in base_predictions.items():
            base_acc = accuracy_score(y, pred_info['predictions'])
            base_accuracies[f'{name}_accuracy'] = base_acc

        result = {
            'ensemble_accuracy': accuracy,
            **base_accuracies
        }

        return result

    def save(self, save_dir: str):
        """保存模型"""
        self.ensemble.save(save_dir)

        # 保存额外配置
        config = {
            'pinyin_vocab_size': self.pinyin_vocab_size,
            'feature_dim': self.feature_dim
        }
        config_filepath = os.path.join(save_dir, "pinyin_ensemble_config.pkl")
        joblib.dump(config, config_filepath)

    def load(self, save_dir: str):
        """加载模型"""
        # 加载配置
        config_filepath = os.path.join(save_dir, "pinyin_ensemble_config.pkl")
        config = joblib.load(config_filepath)

        self.pinyin_vocab_size = config['pinyin_vocab_size']
        self.feature_dim = config['feature_dim']

        # 加载集成模型
        self.ensemble = EnsembleClassifier(n_classes=self.pinyin_vocab_size)
        self.ensemble.load(save_dir)




