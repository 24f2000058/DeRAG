import re
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "main.py")

FAKE_DOCS = [
    {
        "metadata": {"doc_hash": f"hash-{i}"},
        "similarity": 0.9 - i * 0.1,
        "score": 0.8 - i * 0.1,
        "normalized_reputation": 0.5,
    }
    for i in range(3)
]

FAKE_ROWS = {
    doc["metadata"]["doc_hash"]: {
        "doc_hash_short": f"short-{i}",
        "ipfs_cid": f"cid-{i}",
        "publisher": f"pub-{i}",
        "reputation": i,
        "verified": True,
    }
    for i, doc in enumerate(FAKE_DOCS)
}


def test_sidebar_source_ordinals_match_prompt_citation_order():
    """Regression test for the citation<->row correspondence: build_prompt()
    numbers citations [1], [2], [3] by context_docs list position, and the
    sidebar must render "Source [N]" markers in that exact same order. This
    pins the invariant so a future change to the sidebar loop (e.g. sorting
    rows by badge state) can't silently desync the visible marker from the
    citation numbers a viewer is trying to match it against.
    """
    with (
        patch("retrieval.vector_store.init_store", return_value=object()),
        patch("app.ui_helpers.get_context_docs", lambda q, m, collection=None: FAKE_DOCS),
        patch("app.ui_helpers.format_audit_row", lambda h: FAKE_ROWS[h]),
        patch("generation.llm_client.build_prompt") as mock_build_prompt,
        patch("generation.llm_client.generate_streaming", lambda p: iter(["answer"])),
    ):
        mock_build_prompt.return_value = "prompt"
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input[0].set_value("test query")
        at.run(timeout=30)

        assert not at.exception, at.exception

        # The order build_prompt actually received must be the untouched,
        # unsorted context_docs list order.
        prompt_docs = mock_build_prompt.call_args[0][1]
        assert [d["metadata"]["doc_hash"] for d in prompt_docs] == [
            d["metadata"]["doc_hash"] for d in FAKE_DOCS
        ]

    sidebar_markdown = [m.value for m in at.sidebar.get("markdown")]
    row_markdown = [v for v in sidebar_markdown if v.startswith('<div class="derag-audit-row">')]
    assert len(row_markdown) == 3

    for ordinal, (row_html, doc) in enumerate(zip(row_markdown, FAKE_DOCS), start=1):
        assert f"Source [{ordinal}]" in row_html
        expected_row = FAKE_ROWS[doc["metadata"]["doc_hash"]]
        assert expected_row["doc_hash_short"] in row_html

        # The marker must be the first thing in the row, not just present
        # somewhere in it, so it stays legible as a per-row label.
        assert re.match(r'^<div class="derag-audit-row"><div class="derag-source-marker">', row_html)
