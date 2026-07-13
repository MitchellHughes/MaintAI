from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from gensim.models import FastText
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


@dataclass
class SemanticSimilarityConfig:
    embedding_dim: int = 100
    window: int = 5
    min_count: int = 2
    workers: int = 4
    sg: int = 1
    epochs: int = 5
    use_sif: bool = True
    sif_a: float = 1e-3
    remove_pc: bool = True
    random_state: int = 42
    n_jobs: int = 4
    show_progress: bool = False
    cwem_augment: bool = True
    cwem_repeats: int = 2
    max_fit_rows: int | None = None


class SemanticSimilarityClassifier:
    """
    Semantic similarity base classifier (CWEM + discourse-style embeddings).

    Trains maintenance-domain word embeddings (FastText, skip-gram context) with
    class-keyword augmentation (CWEM-inspired same/different class separation).
    Record vectors use SIF weighting and common-discourse removal; class vectors are
    averages of keyword embeddings (paper semantic-similarity step).
    """

    def __init__(
        self,
        failure_keywords: Dict[str, list[str]],
        config: SemanticSimilarityConfig | None = None,
    ) -> None:
        self.failure_keywords = failure_keywords
        self.config = config or SemanticSimilarityConfig()
        self.classes = list(failure_keywords.keys())

        self.embedding_model: FastText | None = None
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.class_prototypes: Dict[str, np.ndarray] | None = None
        self.pc: np.ndarray | None = None

    def _tokenize(self, text: str) -> list[str]:
        text = "" if text is None else str(text).lower()
        text = "".join([c if c.isalnum() or c.isspace() else " " for c in text])
        text = " ".join(text.split())
        return text.split()

    def _cwem_augmented_corpus(self, corpus: list[str]) -> list[str]:
        """Augment corpus with within-class keyword phrases (CWEM same-class cohesion)."""
        if not self.config.cwem_augment:
            return corpus
        augmented = list(corpus)
        for _ in range(max(self.config.cwem_repeats, 0)):
            for keywords in self.failure_keywords.values():
                phrase_tokens = []
                for kw in keywords:
                    phrase_tokens.extend(self._tokenize(kw))
                if phrase_tokens:
                    augmented.append(" ".join(phrase_tokens))
        return augmented

    def _sample_corpus(self, corpus: list[str]) -> list[str]:
        max_rows = self.config.max_fit_rows
        if not max_rows or len(corpus) <= max_rows:
            return corpus
        rng = np.random.RandomState(self.config.random_state)
        idx = rng.choice(len(corpus), size=max_rows, replace=False)
        return [corpus[i] for i in idx]

    def _fit_embeddings(self, corpus: Iterable[str]) -> list[list[str]]:
        docs = self._cwem_augmented_corpus(self._sample_corpus(list(corpus)))
        if self.config.show_progress:
            tokenized = [self._tokenize(doc) for doc in tqdm(docs, desc="Tokenizing", total=len(docs))]
        else:
            tokenized = [self._tokenize(doc) for doc in docs]
        self.embedding_model = FastText(
            sentences=tokenized,
            vector_size=self.config.embedding_dim,
            window=self.config.window,
            min_count=self.config.min_count,
            workers=max(1, self.config.workers),
            sg=self.config.sg,
            epochs=max(1, self.config.epochs),
        )
        return tokenized

    def _fit_tfidf(self, corpus: Iterable[str]) -> None:
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            lowercase=True,
            token_pattern=None,
        )
        self.tfidf_vectorizer.fit(corpus)

    def _get_word_weight(self, token: str) -> float:
        if self.tfidf_vectorizer is None:
            return 1.0
        vocab = self.tfidf_vectorizer.vocabulary_
        if token not in vocab:
            return 1.0
        idx = vocab[token]
        idf = float(self.tfidf_vectorizer.idf_[idx])
        if self.config.use_sif:
            return self.config.sif_a / (self.config.sif_a + idf)
        return idf

    def _sentence_vector(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vecs = []
        weights = []
        for token in tokens:
            if token in self.embedding_model.wv:
                vecs.append(self.embedding_model.wv[token])
                weights.append(self._get_word_weight(token))
        if not vecs:
            return np.zeros(self.config.embedding_dim)
        vecs = np.array(vecs)
        weights = np.array(weights).reshape(-1, 1)
        sent_vec = (vecs * weights).sum(axis=0) / max(weights.sum(), 1e-9)
        return sent_vec

    def _compute_sentence_vectors(self, texts: Iterable[str], desc: str) -> np.ndarray:
        texts = list(texts)
        if self.config.n_jobs and self.config.n_jobs > 1:
            vectors = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(
                delayed(self._sentence_vector)(text) for text in texts
            )
            return np.vstack(vectors)

        iterator = texts
        if self.config.show_progress:
            iterator = tqdm(texts, desc=desc, total=len(texts))
        return np.vstack([self._sentence_vector(text) for text in iterator])

    def _remove_first_pc(self, matrix: np.ndarray) -> np.ndarray:
        if not self.config.remove_pc:
            return matrix
        svd = TruncatedSVD(n_components=1, random_state=self.config.random_state)
        svd.fit(matrix)
        self.pc = svd.components_[0]
        return matrix - matrix.dot(self.pc.reshape(-1, 1)) * self.pc

    def _build_class_prototypes(self) -> None:
        prototypes: Dict[str, np.ndarray] = {}
        for label, keywords in self.failure_keywords.items():
            keyword_vecs = []
            for kw in keywords:
                kw_tokens = self._tokenize(kw)
                token_vecs = [
                    self.embedding_model.wv[tok]
                    for tok in kw_tokens
                    if tok in self.embedding_model.wv
                ]
                if token_vecs:
                    keyword_vecs.append(np.mean(token_vecs, axis=0))
            if keyword_vecs:
                prototypes[label] = np.mean(keyword_vecs, axis=0)
            else:
                prototypes[label] = np.zeros(self.config.embedding_dim)
        self.class_prototypes = prototypes

    def fit(self, corpus: Iterable[str]) -> "SemanticSimilarityClassifier":
        corpus_list = list(corpus)
        fit_corpus = self._sample_corpus(corpus_list)
        tokenized = self._fit_embeddings(fit_corpus)
        self._fit_tfidf(fit_corpus)
        self._build_class_prototypes()

        if self.config.remove_pc and len(tokenized) <= 3000:
            texts = [" ".join(tokens) for tokens in tokenized]
            sent_matrix = self._compute_sentence_vectors(texts, desc="SIF vectors")
            _ = self._remove_first_pc(sent_matrix)
        return self

    def _transform_sentence_matrix(self, texts: Iterable[str]) -> np.ndarray:
        sent_vectors = self._compute_sentence_vectors(texts, desc="Transform vectors")
        if self.config.remove_pc and self.pc is not None:
            sent_vectors = sent_vectors - sent_vectors.dot(self.pc.reshape(-1, 1)) * self.pc
        return sent_vectors

    def transform(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> pd.DataFrame:
        if self.embedding_model is None or self.class_prototypes is None:
            raise ValueError("Classifier must be fitted before transforming.")

        texts = df[column_name].fillna("").astype(str).tolist()
        sent_vectors = self._transform_sentence_matrix(texts)
        class_matrix = np.vstack([self.class_prototypes[c] for c in self.classes])
        sent_norm = sent_vectors / (np.linalg.norm(sent_vectors, axis=1, keepdims=True) + 1e-9)
        class_norm = class_matrix / (np.linalg.norm(class_matrix, axis=1, keepdims=True) + 1e-9)
        sim = sent_norm @ class_norm.T
        return pd.DataFrame(sim, columns=self.classes, index=df.index)

    def predict(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> tuple[pd.Series, pd.DataFrame]:
        scores = self.transform(df, column_name)
        preds = scores.idxmax(axis=1)
        return preds, scores
