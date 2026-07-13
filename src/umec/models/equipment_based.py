"""Equipment-based base classifier (paper: part/component tokens + failure prominence weights)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from umec.data.preprocessing import normalize_tokens

_UNCLASSIFIED_LABEL = "unclassified"
_MIN_SCORE = 1e-9


@dataclass
class EquipmentBasedConfig:
    ngram_range: tuple[int, int] = (1, 2)
    lowercase: bool = True
    use_idf: bool = True
    sublinear_tf: bool = True
    normalize_tokens: bool = True
    default_prominence: float = 0.25


class EquipmentBasedClassifier:
    """
    Identifies failed parts/components in a record, then scores failure mechanisms using
    domain prominence weights for each part (weighted sum of part TF-IDF scores).
    """

    def __init__(
        self,
        failure_keywords: Dict[str, list[str]],
        part_keywords: Dict[str, list[str]],
        part_prominence: Dict[str, Dict[str, float]],
        token_map: Dict[str, str] | None = None,
        config: EquipmentBasedConfig | None = None,
    ) -> None:
        self.failure_keywords = failure_keywords
        self.part_keywords = part_keywords
        self.part_prominence = part_prominence
        self.token_map = token_map or {}
        self.config = config or EquipmentBasedConfig()
        self.classes: list[str] = list(failure_keywords.keys())

        self.vectorizer: TfidfVectorizer | None = None
        self.part_to_class_weights: np.ndarray | None = None
        self.part_names: list[str] = []
        self.part_feature_indices: list[np.ndarray] = []

    @classmethod
    def from_resource_files(
        cls,
        failure_keywords: Dict[str, list[str]],
        part_keywords_path: str | Path,
        prominence_path: str | Path,
        token_map: Dict[str, str] | None = None,
        config: EquipmentBasedConfig | None = None,
    ) -> "EquipmentBasedClassifier":
        with open(part_keywords_path, encoding="utf-8") as handle:
            part_keywords = json.load(handle)
        with open(prominence_path, encoding="utf-8") as handle:
            part_prominence = json.load(handle)
        return cls(
            failure_keywords=failure_keywords,
            part_keywords=part_keywords,
            part_prominence=part_prominence,
            token_map=token_map,
            config=config,
        )

    def _normalize_term(self, term: str) -> str:
        term = term.lower().strip()
        if self.config.normalize_tokens:
            term = normalize_tokens(term, self.token_map)
        return term

    def _build_vocabulary(self) -> list[str]:
        tokens = set()
        for phrases in self.part_keywords.values():
            for phrase in phrases:
                norm = self._normalize_term(phrase)
                tokens.add(norm)
                tokens.update(norm.split())
        return sorted(tokens)

    def _build_weight_matrix(self) -> np.ndarray:
        """Matrix (n_parts, n_classes) of prominence weights."""
        n_parts = len(self.part_names)
        n_classes = len(self.classes)
        weights = np.full((n_parts, n_classes), self.config.default_prominence, dtype=float)

        for part_idx, part in enumerate(self.part_names):
            class_weights = self.part_prominence.get(part, {})
            for class_idx, label in enumerate(self.classes):
                if label in class_weights:
                    weights[part_idx, class_idx] = float(class_weights[label])
        return weights

    def fit(self, corpus: Iterable[str]) -> "EquipmentBasedClassifier":
        vocab = self._build_vocabulary()
        if not vocab:
            raise ValueError("Part keyword vocabulary is empty.")

        self.part_names = sorted(self.part_keywords.keys())
        self.vectorizer = TfidfVectorizer(
            vocabulary=vocab,
            ngram_range=self.config.ngram_range,
            lowercase=self.config.lowercase,
            use_idf=self.config.use_idf,
            sublinear_tf=self.config.sublinear_tf,
        )
        self.vectorizer.fit(list(corpus))
        self.part_to_class_weights = self._build_weight_matrix()
        self.part_feature_indices = self._build_part_feature_indices()
        return self

    def _build_part_feature_indices(self) -> list[np.ndarray]:
        """Precompute vectorizer feature indices for each part once at fit time."""
        if self.vectorizer is None:
            return []
        feature_names = self.vectorizer.get_feature_names_out()
        feature_index = {name: idx for idx, name in enumerate(feature_names)}
        out: list[np.ndarray] = []
        for part in self.part_names:
            idxs: set[int] = set()
            for phrase in self.part_keywords.get(part, []):
                norm = self._normalize_term(phrase)
                idx = feature_index.get(norm)
                if idx is not None:
                    idxs.add(idx)
                for token in norm.split():
                    tok_idx = feature_index.get(token)
                    if tok_idx is not None:
                        idxs.add(tok_idx)
            out.append(np.array(sorted(idxs), dtype=int) if idxs else np.array([], dtype=int))
        return out

    def _part_activation(self, tfidf_row) -> np.ndarray:
        """
        Backward-compatible single-row part activation.

        Inference/XAI paths still call this helper for strict-support checks, so keep
        an efficient row-level implementation even though transform() is vectorized.
        """
        activations = np.zeros(len(self.part_names), dtype=float)
        if not self.part_feature_indices:
            return activations
        for part_idx, idxs in enumerate(self.part_feature_indices):
            if idxs.size == 0:
                continue
            col = tfidf_row[:, idxs].max(axis=1)
            value = np.asarray(col.toarray() if hasattr(col, "toarray") else col).ravel()
            activations[part_idx] = float(value[0]) if value.size else 0.0
        return activations

    def transform(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> pd.DataFrame:
        if self.vectorizer is None or self.part_to_class_weights is None:
            raise ValueError("Classifier must be fitted before transforming.")

        texts = df[column_name].fillna("").astype(str)
        if self.config.normalize_tokens and self.token_map:
            texts = texts.apply(lambda x: normalize_tokens(x, self.token_map))

        tfidf = self.vectorizer.transform(texts)
        n_rows = tfidf.shape[0]
        n_parts = len(self.part_names)
        part_act = np.zeros((n_rows, n_parts), dtype=float)
        for part_idx, idxs in enumerate(self.part_feature_indices):
            if idxs.size == 0:
                continue
            # Max TF-IDF over this part's phrase/token features for all rows.
            max_col = tfidf[:, idxs].max(axis=1)
            part_act[:, part_idx] = np.asarray(max_col.toarray() if hasattr(max_col, "toarray") else max_col).ravel()

        scores = part_act @ self.part_to_class_weights

        return pd.DataFrame(scores, columns=self.classes, index=df.index)

    def predict(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> tuple[pd.Series, pd.DataFrame]:
        score_df = self.transform(df, column_name)
        max_scores = score_df.max(axis=1)
        preds = score_df.idxmax(axis=1).astype(str)
        preds = preds.where(max_scores > _MIN_SCORE, _UNCLASSIFIED_LABEL)
        return preds, score_df
