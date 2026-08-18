"""Tests for scripts/ingest.py.

The tests in this file that need no live services (all except the one
marked @pytest.mark.slow) run by default. The slow test requires BOTH:

  - a running Hardhat node on WEB3_PROVIDER_URI (see .env.example),
    deployed via `npx hardhat run scripts/deploy.ts --network localhost`
    from inside contracts/
  - a running `ipfs daemon` reachable at IPFS_API_URL (see .env.example)

Run the fast suite: `uv run pytest tests/test_scripts_ingest.py -v`
Run the slow test:  `uv run pytest -m slow tests/test_scripts_ingest.py -v`
"""

from pathlib import Path

import pytest

from scripts.ingest import _collect_input_files, _extract_text


def test_extract_text_reads_txt_file_as_utf8(tmp_path):
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Solar panels convert sunlight into electricity.", encoding="utf-8")

    assert _extract_text(txt_path) == "Solar panels convert sunlight into electricity."


def test_extract_text_reads_pdf_file(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    pdf_path = tmp_path / "sample.pdf"

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # A blank page extracts to empty (or near-empty) text — this test only
    # proves the .pdf branch runs pypdf's extraction path without raising,
    # not that extraction produces specific content.
    result = _extract_text(pdf_path)
    assert isinstance(result, str)


def test_collect_input_files_single_file_returns_that_file(tmp_path):
    txt_path = tmp_path / "solo.txt"
    txt_path.write_text("content", encoding="utf-8")

    assert _collect_input_files(txt_path) == [txt_path]


def test_collect_input_files_directory_returns_sorted_txt_and_pdf_only(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "d.md").write_text("ignored", encoding="utf-8")

    result = _collect_input_files(tmp_path)

    assert [p.name for p in result] == ["a.txt", "b.txt", "c.pdf"]
