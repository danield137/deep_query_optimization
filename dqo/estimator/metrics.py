from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, cast, List

import numpy as np
import sklearn.metrics as skm


def mcc_metrics(true, pred) -> Dict[str, float]:
    return {
        'accuracy': skm.accuracy_score(true, pred),
        'balanced accuracy': skm.balanced_accuracy_score(true, pred, adjusted=True),
        'kappa': skm.cohen_kappa_score(true, pred),
        'recall': skm.recall_score(true, pred, average='macro', zero_division=0),
        'f1 macro': skm.f1_score(true, pred, average='macro', zero_division=0),
        'f1 weighted': skm.f1_score(true, pred, average='weighted', zero_division=0)
    }


def regression_metrics(true, pred) -> Dict[str, float]:
    if not isinstance(true, np.ndarray):
        true = np.array(true)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred) 
    return {
        'mae': cast(float, np.mean(np.abs(pred - true)))
    }


@dataclass
class TwoSidedError:
    left: List[float] = field(default_factory=list)
    right: List[float] = field(default_factory=list)
    hits: int = field(default_factory=int)

    def record_error(self, p, t):
        err = p - t
        rounded_err = np.round(p) - np.round(t)
        if err > 0:
            self.right.append(err)
        elif err < 0:
            self.left.append(err)

        if rounded_err == 0:
            self.hits += 1

    def both(self):
        return self.left + self.right

    def mean(self):
        return np.nan_to_num(np.mean(self.left)), np.nan_to_num(np.mean(self.right))

    def accuracy(self):
        return self.hits / (self.hits + len(self.left) + len(self.right)) if self.right or self.left else 0


def custom_metrics(true, pred):
    errors: Dict[int, TwoSidedError] = defaultdict(TwoSidedError)

    for t, p in zip(true, pred):
        t_rounded = int(np.round(t))
        errors[t_rounded].record_error(p, t)

    mean_rounded = []
    bucket_errors = []
    bucket_accuracy = []
    values = sorted(errors.keys())
    for bucket in values:
        mean_rounded.append(errors[bucket].mean())
        h, _ = np.histogram(errors[bucket].both(), bins=range(10))
        bucket_errors.append(h.tolist())
        bucket_accuracy.append(errors[bucket].accuracy())

    return {
        'mean_rounded_two_sided_error': mean_rounded,
        'bucket_errors': bucket_errors,
        'bucket_accuracy': bucket_accuracy,
        'values': values
    }
