from rag.llm import truncate_chunks_to_fit


def test_truncate_chunks_truncates_when_exceeding_limit():
    # Available tokens: max_tokens - reserved (instruction_tokens=500)
    chunks = ["a" * 400, "b" * 400]

    truncated = truncate_chunks_to_fit(
        chunks=chunks,
        max_tokens=600,  # leaves ~100 tokens for context
        query="",
        system_prompt="",
        conversation_history=None,
    )

    assert len(truncated) == 1  # should stop after truncating first chunk
    assert truncated[0] != chunks[0]  # content was truncated
    assert "[... phần còn lại đã được cắt bớt" in truncated[0]
    # Ensure the truncated chunk is shorter than original
    assert len(truncated[0]) < len(chunks[0])


def test_truncate_chunks_keeps_multiple_when_space_available():
    chunks = ["alpha" * 10, "beta" * 10]

    truncated = truncate_chunks_to_fit(
        chunks=chunks,
        max_tokens=900,  # plenty of room after reserved tokens
        query="",
        system_prompt="",
        conversation_history=None,
    )

    assert truncated == chunks  # unchanged when within limits

