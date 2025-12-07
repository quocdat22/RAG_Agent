from rag.retriever import (
    CandidateChunk,
    _reciprocal_rank_fusion,
    _weighted_sum_fusion,
)


def test_reciprocal_rank_fusion_prioritizes_overlap():
    dense = [
        CandidateChunk(text="a", metadata={"from": "dense"}, score=0.9),
        CandidateChunk(text="b", metadata={"from": "dense"}, score=0.8),
    ]
    sparse = [
        CandidateChunk(text="b", metadata={"from": "sparse"}, score=1.0),
        CandidateChunk(text="c", metadata={"from": "sparse"}, score=0.5),
    ]

    fused = _reciprocal_rank_fusion(
        dense_results=dense,
        sparse_results=sparse,
        dense_weight=2.0,  # favor dense contributions to break ties
        sparse_weight=1.0,
    )

    assert [c.text for c in fused[:3]] == ["b", "a", "c"]
    # Metadata should prefer dense version when available
    assert fused[0].metadata["from"] == "dense"


def test_weighted_sum_fusion_normalizes_and_combines_scores():
    dense = [
        CandidateChunk(text="a", metadata={}, score=0.0),
        CandidateChunk(text="b", metadata={"from_dense": True}, score=1.0),
    ]
    sparse = [
        CandidateChunk(text="b", metadata={"from_sparse": True}, score=0.0),
        CandidateChunk(text="c", metadata={"from_sparse": True}, score=0.5),
    ]

    fused = _weighted_sum_fusion(
        dense_results=dense,
        sparse_results=sparse,
        dense_weight=0.7,
        sparse_weight=0.3,
    )

    assert [c.text for c in fused] == ["b", "c", "a"]
    # When both sources exist, prefer metadata from dense result
    assert fused[0].metadata["from_dense"] is True

