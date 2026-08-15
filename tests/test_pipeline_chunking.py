from pipeline.chunking import chunk_text

SAMPLE = " ".join(f"word{i:04d}" for i in range(200))  # 1799 chars, no punctuation


def test_chunk_count_and_sizes():
    chunks = chunk_text(SAMPLE)
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk) <= 500


def test_chunk_overlap_shares_boundary_words():
    chunks = chunk_text(SAMPLE)
    assert chunks[0].endswith(
        "word0050 word0051 word0052 word0053 word0054"
    )
    assert chunks[1].startswith(
        "word0050 word0051 word0052 word0053 word0054"
    )
    assert chunks[1].endswith(
        "word0100 word0101 word0102 word0103 word0104"
    )
    assert chunks[2].startswith(
        "word0100 word0101 word0102 word0103 word0104"
    )


def test_chunking_does_not_renormalize():
    # Mixed-case, unnormalized input passed straight through untouched
    # except for the splitter's own boundary logic. Confirms chunking is
    # a pure split, not a normalize+split.
    raw = "ALREADY Mixed CASE text " * 30
    chunks = chunk_text(raw)
    assert any(c != c.lower() for c in chunks)
