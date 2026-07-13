from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from umec.data.preprocessing import normalize_tokens

_UNCLASSIFIED_LABEL = "unclassified"
_MIN_SCORE = 1e-9


@dataclass
class TokenMatchingConfig:
    ngram_range: tuple[int, int] = (1, 2)
    lowercase: bool = True
    use_idf: bool = True
    sublinear_tf: bool = True
    normalize_tokens: bool = True


class TokenMatchingClassifier:
    """
    Token matching base classifier (ISO/SME failure-mechanism vocabulary).

    For each record, TF-IDF weights are computed for domain tokens; the score for a
    failure-mechanism class is the **sum** of TF-IDF values of tokens linked to that class.
    The predicted class is the argmax over class sums (paper Eq. token-matching).
    """

    def __init__(
        self,
        failure_keywords: Dict[str, list[str]],
        token_map: Dict[str, str] | None = None,
        config: TokenMatchingConfig | None = None,
    ) -> None:
        self.failure_keywords = failure_keywords
        self.token_map = token_map or {}
        self.config = config or TokenMatchingConfig()

        self.classes: list[str] = list(failure_keywords.keys())
        self.vectorizer: TfidfVectorizer | None = None
        self.mapping_matrix: csr_matrix | None = None
        self.feature_names: list[str] | None = None
        self.feature_index: dict[str, int] = {}
        self.phrase_feature_indices: dict[int, list[int]] = {}

    def _normalize_keyword(self, keyword: str) -> str:
        keyword = keyword.lower().strip()
        if self.config.normalize_tokens:
            keyword = normalize_tokens(keyword, self.token_map)
        return keyword

    def fit(self, corpus: Iterable[str]) -> "TokenMatchingClassifier":
        tokens = sorted(
            {
                self._normalize_keyword(t)
                for sublist in self.failure_keywords.values()
                for t in sublist
            }
        )

        if not tokens:
            raise ValueError("Failure keywords are empty; cannot build vocabulary.")

        self.vectorizer = TfidfVectorizer(
            vocabulary=tokens,
            ngram_range=self.config.ngram_range,
            lowercase=self.config.lowercase,
            use_idf=self.config.use_idf,
            sublinear_tf=self.config.sublinear_tf,
        )
        self.vectorizer.fit(corpus)
        self.feature_names = list(self.vectorizer.get_feature_names_out())
        self.feature_index = {token: idx for idx, token in enumerate(self.feature_names)}
        self.phrase_feature_indices = {}

        mapping_matrix = np.zeros((len(self.feature_names), len(self.classes)))

        for class_idx, label in enumerate(self.classes):
            phrase_idxs: list[int] = []
            for token in self.failure_keywords[label]:
                token = self._normalize_keyword(token)
                idx = self.feature_index.get(token)
                if idx is not None:
                    # Multi-word phrases are stronger evidence than single tokens.
                    weight = 2.0 if " " in token else 1.0
                    mapping_matrix[idx, class_idx] = weight
                    if " " in token:
                        phrase_idxs.append(idx)
            self.phrase_feature_indices[class_idx] = phrase_idxs

        self.mapping_matrix = csr_matrix(mapping_matrix)
        return self

    def transform(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> pd.DataFrame:
        if self.vectorizer is None or self.mapping_matrix is None:
            raise ValueError("Classifier must be fitted before transforming.")

        texts = df[column_name].fillna("").astype(str)
        if self.config.normalize_tokens and self.token_map:
            texts = texts.apply(lambda x: normalize_tokens(x, self.token_map))

        # Positional indexing only — sparse matrices reject pandas Series masks.
        texts = texts.reset_index(drop=True)
        x_mat = self.vectorizer.transform(texts)
        raw_scores = x_mat.dot(self.mapping_matrix).toarray().astype(np.float64)

        # Phrase boost from sparse columns only (avoids per-phrase regex scans across all rows).
        for class_idx, phrase_idxs in self.phrase_feature_indices.items():
            if not phrase_idxs:
                continue
            phrase_weights = np.asarray(x_mat[:, phrase_idxs].sum(axis=1)).ravel()
            if phrase_weights.size:
                raw_scores[:, class_idx] += phrase_weights * 0.5

        # No vocabulary hit in this row → all-zero scores (do not argmax to first class).
        hit_counts = np.asarray(x_mat.sum(axis=1)).ravel()
        for row_idx, hits in enumerate(hit_counts):
            if float(hits) <= _MIN_SCORE:
                raw_scores[row_idx, :] = 0.0
        return pd.DataFrame(raw_scores, columns=self.classes, index=df.index)

    def predict(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> tuple[pd.Series, pd.DataFrame]:
        scores = self.transform(df, column_name)
        max_scores = scores.max(axis=1)
        preds = scores.idxmax(axis=1).astype(str)
        preds = preds.where(max_scores > _MIN_SCORE, _UNCLASSIFIED_LABEL)
        return preds, scores
