from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from umec.models.base import BaseUnsupervisedClassifier
from umec.models.spectral import (
    SpectralBasis,
    fit_spectral_basis,
    project_spectral,
    rank_difference_features,
)


@dataclass
class UMECConfig:
    ecoc_scheme: str = "pairwise"
    aggregation: str = "spectral"
    prior_weight: float = 0.35
    allow_unclassified: bool = True
    unclassified_threshold: float = 0.0
    use_spectral: bool = True
    spectral_components: int = 3
    use_third_moment: bool = True
    # When set, spectral/ECOC calibration uses a random subset (avoids OOM on huge uploads).
    fit_sample_rows: int = 0
    predict_chunk_size: int = 5000


class UMECClassifier:
    """
    Unsupervised Multi-class Ensemble Classifier (UMEC).

    Pipeline (paper-aligned):
    1. Multiple unsupervised base classifiers produce continuous class scores.
    2. ECOC pairwise encoding with max order-statistic reduction (pos_max - neg_max).
    3. Ranked differences of reduction statistics + spectral decomposition of 2nd/3rd moments.
    4. Prior-adjusted ECOC decoding for class imbalance.
    """

    def __init__(
        self,
        classifiers: List[BaseUnsupervisedClassifier],
        classes: list[str] | None = None,
        ecoc_matrix: np.ndarray | None = None,
        config: UMECConfig | None = None,
    ) -> None:
        self.classifiers = classifiers
        self.classes = classes
        self.ecoc_matrix = ecoc_matrix
        self.config = config or UMECConfig()

        self.bit_labels: list[str] = []
        self.class_priors: pd.Series | None = None
        self.spectral_bases: dict[int, SpectralBasis] = {}

    def _score_df(self, clf: BaseUnsupervisedClassifier, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        out = clf.predict(df, column_name=column_name)
        scores = out[1] if isinstance(out, tuple) else out
        if not isinstance(scores, pd.DataFrame):
            scores = pd.DataFrame(scores, index=df.index, columns=clf.classes)
        return scores

    def _common_classes(self, score_dfs: list[pd.DataFrame]) -> list[str]:
        common = set(score_dfs[0].columns)
        for s in score_dfs[1:]:
            common &= set(s.columns)
        if not common:
            raise ValueError("No shared class columns across classifiers.")
        return sorted(common)

    def _build_pairwise_ecoc(self, classes: list[str]) -> np.ndarray:
        bits = []
        labels = []
        for i, j in combinations(range(len(classes)), 2):
            col = np.zeros(len(classes))
            col[i] = 1
            col[j] = -1
            bits.append(col)
            labels.append(f"{classes[i]}_vs_{classes[j]}")
        self.bit_labels = labels
        return np.stack(bits, axis=1)

    def _resolve_ecoc(self, classes: list[str]) -> np.ndarray:
        if self.ecoc_matrix is not None:
            if self.ecoc_matrix.shape[0] != len(classes):
                raise ValueError("ECOC matrix row count must match number of classes.")
            return self.ecoc_matrix

        if self.config.ecoc_scheme == "pairwise":
            return self._build_pairwise_ecoc(classes)

        raise ValueError(f"Unsupported ECOC scheme: {self.config.ecoc_scheme}")

    def _reduction_stats(self, score_df: pd.DataFrame) -> np.ndarray:
        if self.ecoc_matrix is None:
            raise ValueError("ECOC matrix is not initialized. Call fit() first.")

        scores = score_df[self.classes].values
        stats = []
        for bit_idx in range(self.ecoc_matrix.shape[1]):
            code = self.ecoc_matrix[:, bit_idx]
            pos_idx = np.where(code == 1)[0]
            neg_idx = np.where(code == -1)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                stat = np.zeros(scores.shape[0])
            else:
                pos_max = scores[:, pos_idx].max(axis=1)
                neg_max = scores[:, neg_idx].max(axis=1)
                stat = pos_max - neg_max
            stats.append(stat)

        return np.stack(stats, axis=1)

    def _stack_reductions(self, aligned_scores: list[pd.DataFrame]) -> np.ndarray:
        """Shape (n_classifiers, n_samples, n_bits)."""
        per_clf = [self._reduction_stats(scores) for scores in aligned_scores]
        return np.stack(per_clf, axis=0)

    def _fit_spectral_bases(self, reduction_stack: np.ndarray) -> None:
        n_bits = reduction_stack.shape[2]
        self.spectral_bases = {}
        for bit_idx in range(n_bits):
            per_sample = reduction_stack[:, :, bit_idx].T
            features = rank_difference_features(per_sample)
            self.spectral_bases[bit_idx] = fit_spectral_basis(
                features,
                n_components=self.config.spectral_components,
                use_third_moment=self.config.use_third_moment,
            )

    def _aggregate_bits(self, reduction_stack: np.ndarray) -> np.ndarray:
        n_bits = reduction_stack.shape[2]
        n_samples = reduction_stack.shape[1]
        bit_scores = np.zeros((n_samples, n_bits), dtype=float)

        use_spectral = self.config.use_spectral and self.config.aggregation == "spectral"
        if use_spectral and self.spectral_bases:
            for bit_idx in range(n_bits):
                per_sample = reduction_stack[:, :, bit_idx].T
                features = rank_difference_features(per_sample)
                basis = self.spectral_bases[bit_idx]
                projected = project_spectral(features, basis)
                if projected.ndim == 1:
                    projected = projected.reshape(-1, 1)
                weights = basis.eigenvalues[: projected.shape[1]]
                weights = weights / max(weights.sum(), 1e-9)
                bit_scores[:, bit_idx] = (projected * weights).sum(axis=1)
            return bit_scores

        if self.config.aggregation == "sum":
            return reduction_stack.sum(axis=0)
        return reduction_stack.mean(axis=0)

    def _fit_sample_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        max_rows = int(self.config.fit_sample_rows or 0)
        if max_rows <= 0 or len(df) <= max_rows:
            return df
        return df.sample(n=max_rows, random_state=42)

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None, column_name: str = "processed_discrepancy") -> "UMECClassifier":
        fit_df = self._fit_sample_frame(df)
        score_dfs = [self._score_df(clf, fit_df, column_name) for clf in self.classifiers]
        common_classes = self._common_classes(score_dfs)
        self.classes = self.classes or common_classes

        if y is not None:
            y = y.astype(str)
            self.class_priors = y.value_counts(normalize=True).reindex(self.classes).fillna(1e-6)
        else:
            self.class_priors = pd.Series(1.0 / len(self.classes), index=self.classes)

        self.ecoc_matrix = self._resolve_ecoc(self.classes)

        aligned = [s[self.classes] for s in score_dfs]
        reduction_stack = self._stack_reductions(aligned)
        if self.config.use_spectral and self.config.aggregation == "spectral":
            self._fit_spectral_bases(reduction_stack)

        return self

    def transform(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> pd.DataFrame:
        score_dfs = [self._score_df(clf, df, column_name) for clf in self.classifiers]
        aligned_scores = [s[self.classes] for s in score_dfs]
        reduction_stack = self._stack_reductions(aligned_scores)
        agg = self._aggregate_bits(reduction_stack)

        bit_labels = self.bit_labels or [f"bit_{i}" for i in range(agg.shape[1])]
        return pd.DataFrame(agg, columns=bit_labels, index=df.index)

    def class_score_df(self, reduction_df: pd.DataFrame) -> pd.DataFrame:
        if self.ecoc_matrix is None:
            raise ValueError("ECOC matrix is not initialized. Call fit() first.")

        reduction = reduction_df.values
        code = self.ecoc_matrix
        denom = (code != 0).sum(axis=1)
        denom = np.where(denom == 0, 1, denom)
        margins = (reduction @ code.T) / denom

        priors = self.class_priors.reindex(self.classes).fillna(1e-6).values
        prior_adj = self.config.prior_weight * np.log(priors + 1e-9)
        scores = margins + prior_adj

        return pd.DataFrame(scores, columns=self.classes, index=reduction_df.index)

    def _decode(self, class_scores: pd.Series) -> tuple[str, float]:
        best_label = class_scores.idxmax()
        best_score = float(class_scores.max())
        return best_label, best_score

    def _predict_chunk(self, df: pd.DataFrame, column_name: str) -> tuple[pd.Series, pd.DataFrame]:
        reduction_df = self.transform(df, column_name)
        class_scores = self.class_score_df(reduction_df)

        labels = []
        for _, row in class_scores.iterrows():
            label, score = self._decode(row)
            if self.config.allow_unclassified and score < self.config.unclassified_threshold:
                labels.append("unclassified")
            else:
                labels.append(label)

        return pd.Series(labels, index=df.index), reduction_df

    def predict(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> tuple[pd.Series, pd.DataFrame]:
        chunk_size = int(self.config.predict_chunk_size or 0)
        if chunk_size <= 0 or len(df) <= chunk_size:
            return self._predict_chunk(df, column_name)

        label_parts: list[pd.Series] = []
        reduction_parts: list[pd.DataFrame] = []
        total_chunks = max(1, (len(df) + chunk_size - 1) // chunk_size)
        logger.info(
            "UMEC internal chunked predict: %s rows, %s chunks of %s",
            f"{len(df):,}",
            total_chunks,
            f"{chunk_size:,}",
        )
        for chunk_index, start in enumerate(range(0, len(df), chunk_size), start=1):
            end = min(start + chunk_size, len(df))
            logger.info(
                "UMEC internal chunk %s/%s — rows %s–%s",
                chunk_index,
                total_chunks,
                f"{start + 1:,}",
                f"{end:,}",
            )
            chunk = df.iloc[start:end]
            labels, reduction = self._predict_chunk(chunk, column_name)
            label_parts.append(labels)
            reduction_parts.append(reduction)

        return pd.concat(label_parts), pd.concat(reduction_parts)
