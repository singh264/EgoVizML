"""This module contains functions for evaluating sklearn models."""

import warnings
import logging
from typing import Protocol
from dataclasses import dataclass

import joblib
import pandas as pd
import polars as pl
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Classifier(Protocol):
    """Protocol for sklearn classifiers."""

    classes_: list

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...

    def predict_proba(self, X):
        ...


@dataclass
class Results:
    """Dataclass for storing results."""

    clf: str
    result: pd.DataFrame
    cm: pd.DataFrame
    auc: float
    samples: pd.DataFrame


def get_preds(clf: Classifier, X_test):
    """Get predictions from a classifier."""
    return clf.predict(X_test)


def preds_dataframe(X_test, y_test, y_pred):
    """Get predictions from a classifier."""
    df = X_test.copy()
    df["y_true"] = y_test
    df["y_pred"] = y_pred
    return df


def evaluate_model(clf: Classifier, X_test, y_test):
    """Evaluate a classifier."""
    y_pred = get_preds(clf, X_test)

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    preds_df = preds_dataframe(X_test, y_test, y_pred)

    return report, matrix, preds_df


def logocv(df, X, y, groups, clf: Classifier, label_encoder):
    """Evaluate a classifier using leave one group out cross-validation."""

    logo = LeaveOneGroupOut()
    evaluation_metrics = {}
    samples = {}
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for train_index, test_index in logo.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # get key as group left out
        group_left_out = df.iloc[test_index]["video"].values[0][:5]

        # initialize and train the classifier
        clf.fit(X_train, y_train)

        # make predictions and evaluate the model
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=1
            )
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)
        accuracy = accuracy_score(y_test, y_pred)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        # save results in a dict
        evaluation_metrics[group_left_out] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        # create samples dict
        samples[group_left_out] = {
            "videos": df.iloc[test_index]["video"].values,
            "X_test": X_test,
            "y_test_label": label_encoder.inverse_transform(y_test),
            "y_pred_label": label_encoder.inverse_transform(y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
        }

    # get confusion matrix and results
    cm = confusion_matrix(all_y_true, all_y_pred)
    auc = roc_auc_score(all_y_true, all_y_prob, multi_class="ovr", average="weighted")
    results = create_results_df(evaluation_metrics, clf)
    samples = pd.DataFrame.from_dict(samples, orient="index")

    # log complete
    logging.info("LOGOCV complete for %s", clf.__class__.__name__)

    return results, cm, auc, samples


def create_results_df(evaluation_metrics: dict, clf: Classifier) -> pd.DataFrame:
    results = pd.DataFrame.from_dict(evaluation_metrics, orient="index")
    results = results.reset_index()
    results = results.rename(columns={"index": "group_left_out"})

    # get the median of the results
    results["median_accuracy"] = results["accuracy"].median()
    results["median_precision"] = results["precision"].median()
    results["median_recall"] = results["recall"].median()
    results["median_f1"] = results["f1"].median()

    # mean of the results
    results["mean_accuracy"] = results["accuracy"].mean()
    results["mean_precision"] = results["precision"].mean()
    results["mean_recall"] = results["recall"].mean()
    results["mean_f1"] = results["f1"].mean()

    # std of the results
    results["std_accuracy"] = results["accuracy"].std()
    results["std_precision"] = results["precision"].std()
    results["std_recall"] = results["recall"].std()
    results["std_f1"] = results["f1"].std()

    # add model name
    results["model"] = clf.__class__.__name__

    return results


def evaluate_models(models, df, label_encoder) -> tuple[list[Results], pd.DataFrame]:
    """Evaluate a list of models using LOGOCV."""
    X = df.drop(["adl", "video"], axis=1)
    y = df["adl"]
    y_encoded = label_encoder.fit_transform(y)
    groups = df["video"].str[:5]

    results: list[Results] = []

    for name, clf in models:
        result = logocv(df, X, y_encoded, groups, clf, label_encoder)
        results.append(Results(name, *result))

    # concat result[0] for result in results.values()
    results_df = pd.concat([result.result for result in results])

    return results, results_df


def display_median_table(results_df) -> pd.DataFrame:
    return (
        results_df[
            [
                "median_accuracy",
                "median_precision",
                "median_recall",
                "median_f1",
                "model",
            ]
        ]
        .groupby("model")
        .first()
        .reset_index()
    )


def display_mean_table(results_df) -> pd.DataFrame:
    return (
        results_df[
            [
                "mean_accuracy",
                "mean_precision",
                "mean_recall",
                "mean_f1",
                "model",
            ]
        ]
        .groupby("model")
        .first()
        .reset_index()
    )


def display_pct_table(results_df, threshold=0.5) -> pd.DataFrame:
    return (
        results_df[["f1", "median_f1", "mean_f1", "std_f1", "model"]]
        .groupby("model")
        .agg(
            median_f1=("median_f1", "first"),
            mean_f1=("mean_f1", "first"),
            std_f1=("std_f1", "first"),
            # get the percentage of f1 scores that are above a threshold
            percentage_above_05=(
                "f1",
                lambda x: round(len(x[x > threshold]) / len(x), 2),
            ),
        )
        .reset_index()
        .rename(columns={"percentage_above_05": f"pct_above_{threshold}"})
    )
