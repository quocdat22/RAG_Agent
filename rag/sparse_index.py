from typing import Iterable, List

from rank_bm25 import BM25Okapi


class SparseIndex:
    def __init__(self, texts: Iterable[str]) -> None:
        self.texts: List[str] = list(texts)
        tokenized_corpus = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 20) -> List[int]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return ranked_indices[:k]


