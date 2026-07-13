from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: Iterable[str],
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (12, 10),
    save_path: str | None = None,
) -> None:
    label_list = list(labels)
    present = set(y_true)
    label_list = [lbl for lbl in label_list if lbl in present]
    if not label_list:
        raise ValueError("No labels present in y_true for confusion matrix.")

    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    n = len(label_list)
    fig, ax = plt.subplots(figsize=figsize, dpi=180)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=90 if n > 10 else 45)
    ax.tick_params(axis="both", labelsize=max(6, 11 - n // 4))
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
