"""
utils.py - 모든 노트북이 공유하는 '채점자 + 기록원'.

직접 실행하지 않습니다. 각 노트북 맨 위에서
    from utils import set_seed, compute_metrics, log_result, load_processed
처럼 불러 씁니다.
목적: 모든 노트북이 똑같은 방식으로 점수를 재고, 한 표(results.csv)에 기록.
"""
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import recall_score, precision_score, f1_score


def set_seed(seed=42):
    """난수를 고정해서 매번 같은 결과(특히 같은 train/val 분할)가 나오게 한다."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def compute_metrics(y_true, y_prob, threshold=0.5):
    """불균형 분류 채점. 주지표는 PR_AUC. y_prob = 백오더일 확률(0~1)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pred = (y_prob >= threshold).astype(int)
    return {
        "PR_AUC": round(average_precision_score(y_true, y_prob), 4),
        "ROC_AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, pred, zero_division=0), 4),
        "threshold": threshold,
    }


def log_result(model_name, metrics, path="results.csv"):
    """results.csv에 한 줄 추가. 08번이 이걸 읽어 전체 모델을 비교."""
    row = {"model": model_name}
    row.update(metrics)
    header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)
    return row


def load_processed(out_dir):
    """02_Preprocess가 저장한 train/val/test를 불러온다 (모두 같은 데이터)."""
    train = pd.read_pickle(os.path.join(out_dir, "train.pkl"))
    val = pd.read_pickle(os.path.join(out_dir, "val.pkl"))
    test = pd.read_pickle(os.path.join(out_dir, "test.pkl"))
    return train, val, test
