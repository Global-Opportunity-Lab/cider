from __future__ import annotations
from autogluon.tabular import TabularPredictor  # type: ignore[import]
from helpers.utils import make_dir, strictly_increasing
from joblib import load  # type: ignore[import]
import numpy as np
from numpy import ndarray
import os
import pandas as pd
from pandas import DataFrame as PandasDataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin, clone  # type: ignore[import]
from sklearn.metrics import confusion_matrix, auc, r2_score  # type: ignore[import]
from pathlib import Path
from typing import Optional, Tuple, Union


def metrics(a1: Union[ndarray, Series], a2: Union[ndarray, Series], p: float) -> Tuple[float, float, float, float, float]:
    """
    Computes classification metrics at a certain threshold, i.e. turn regression into classification by considering the
    bottom p% of targets as belonging to the positive class.

    Args:
        a1: Vector of true values.
        a2: Vector of predicted values.
        p: Classification threshold.

    Returns: Tuple with accuracy, precision, recall, true positive rate, false positive rate.
    """
    if p == 0 or p == 100:
        raise ValueError('Percentage targeting must be between 0 and 100 (exclusive).')

    num_ones = int((p/100)*len(a1))
    num_zeros = len(a1) - num_ones
    targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
    
    a = np.vstack([a1, a2])
    a = a[:, a[0, :].argsort()]
    a[0, :] = targeting_vector
    a = a[:, a[1, :].argsort()]
    a[1, :] = targeting_vector
    
    tn, fp, fn, tp = confusion_matrix(a[0, :], a[1, :]).ravel()

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    tpr = recall
    fpr = fp/(fp + tn)

    return accuracy, precision, recall, tpr, fpr


def auc_overall(a1: Union[ndarray, Series], a2: Union[ndarray, Series]) -> float:
    """
    Computes AUC score by turning regression problem into classification using 'metrics' function.

    Args:
        a1: Vector of true values.
        a2: Vector of predicted values.

    Returns: AUC score.
    """
    grid = np.linspace(1, 100, 99)[:-1]
    metrics_grid = [metrics(a1, a2, p) for p in grid]
    tprs, fprs = [g[3] for g in metrics_grid], [g[4] for g in metrics_grid]
    
    fprs[0] = 0
    tprs[0] = 0
    fprs.append(1)
    tprs.append(1)
    
    while not strictly_increasing(fprs):
        to_remove = []
        for j in range(1, len(fprs)):
            if fprs[j] <= fprs[j-1]:
                to_remove.append(j)
        fprs = [fprs[i] for i in range(len(fprs)) if i not in to_remove]
        tprs = [tprs[i] for i in range(len(tprs)) if i not in to_remove]

    return auc(fprs, tprs)


class DropMissing(TransformerMixin, BaseEstimator):

    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold = threshold

    def fit(self, X: PandasDataFrame, y: Optional[Series] = None) -> DropMissing:
        missing = X.isna().mean()
        self.missing_frac = missing
        self.cols_to_drop = missing[missing > self.threshold].index
        self.cols_to_keep = missing[missing <= self.threshold].index
        return self

    def transform(self, X: PandasDataFrame, y: Optional[Series] = None) -> PandasDataFrame:
        return X.drop(self.cols_to_drop.values, axis=1)


class Winsorizer(TransformerMixin, BaseEstimator):
    def __init__(self, limits: Optional[Tuple[float, float]] = None) -> None:
        self.limits = limits

    def fit(self, X: PandasDataFrame, y: Optional[Series] = None) -> Winsorizer:
        X = pd.DataFrame(X)
        if self.limits is None:
            self.limits = (0.01, 0.99)
        elif isinstance(self.limits, float):
            self.limits = (self.limits, 1 - self.limits)

        columns = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        threshold_dict = {}

        for column in columns:
            low, high = X[column].quantile(self.limits)
            threshold_dict[column] = (low, high)

        self.columns_ = columns
        self.threshold_dict_ = threshold_dict

        return self

    def transform(self, X: Union[PandasDataFrame, ndarray], y: Optional[Series] = None) -> PandasDataFrame:
        X = pd.DataFrame(X)
        X_t = X.copy()

        def trim(x: Union[float, int], low: Union[float, int], high: Union[float, int]) -> Union[float, int]:
            if pd.isna(x):
                return x
            else:
                x = low if x < low else x
                x = high if x > high else x
                return x
        trim_vec = np.vectorize(trim)

        for column, tup in self.threshold_dict_.items():
            X_t[column] = trim_vec(X_t[column], *tup)

        return X_t


def load_model(model: str, out_path: Path, kind: str = 'tuned'):
    """
    Loads trained ML model. If tuned, the best performing model will be loaded.

    Args:
        model: The name of the model to load.
        out_path: The path to the folder where models are saved.
        kind: The type of model, i.e. untuned, tuned, or automl.

    Returns: The loaded model.
    """
    subdir = kind + '_models'
    
    full_path = out_path / subdir / model / 'model'

    if full_path.is_file():
        model_name = model
        model = load(full_path)
    elif full_path.is_dir():
        model_name = model
        model = TabularPredictor.load(full_path)
    elif os.path.isfile(model):
        model_name = model.split('/')[-1]
        model = load(model)
        make_dir(out_path / subdir / model_name)
    else:
        raise ValueError("The 'model' argument should be a path or a recognized model name")

    if kind == 'tuned':
        model = model.best_estimator_

    return model_name, model
