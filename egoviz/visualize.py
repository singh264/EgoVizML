"""This module contains functions for visualizing sklearn models or images."""

from typing import Protocol
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import numpy as np

from egoviz.models.evaluation import Classifier


class LabelEncoder(Protocol):
    """Protocol for sklearn LabelEncoders."""

    def fit_transform(self, y): ...

    def inverse_transform(self, y): ...


def plot_cm(
    cm,
    clf: Classifier,
    label_encoder: LabelEncoder | None = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    figsize=(8, 6),
    dpi=600,
    annot_font_size=14,  # Add an argument for annotation font size
):
    """Plot a confusion matrix with the option to normalize."""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    df_cm = pd.DataFrame(
        cm,
        index=(
            label_encoder.inverse_transform(clf.classes_)
            if label_encoder
            else clf.classes_
        ),
        columns=(
            label_encoder.inverse_transform(clf.classes_)
            if label_encoder
            else clf.classes_
        ),
    )
    fig = plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(df_cm, annot=True, fmt=fmt, cmap="Blues", annot_kws={"size": annot_font_size})
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def draw_boxes(
    img_path: str,
    boxes: list[list[int]],
    labels: list[str],
    active: list[bool],
    save_path: str | None = None,
):
    """Draw bounding boxes on an image."""

    # read image
    img = plt.imread(img_path)

    # draw boxes
    for box, label, act in zip(boxes, labels, active):
        color = (0, 255, 0) if act else (255, 0, 0)
        img = draw_box(img, box, label, color=color)

    # save image
    if save_path:
        plt.imsave(save_path, img)

    # show image without axis
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def draw_box(img, box, label, color=(0, 255, 0)):
    """Draw a single bounding box on an image."""
    x1, y1, x2, y2 = box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
    img = cv2.putText(
        img,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color=color,
        thickness=2,
    )
    return img
