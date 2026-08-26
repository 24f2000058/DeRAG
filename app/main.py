import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb.errors
import huggingface_hub.errors
import streamlit as st

from app.ui_helpers import DERAG_MODE, STANDARD_RAG_MODE, format_audit_row, get_context_docs
from blockchain.registry_client import RegistryError
from generation.llm_client import GenerationError, build_prompt, generate_streaming
from retrieval.vector_store import init_store

st.set_page_config(page_title="DeRAG", page_icon="🔗")

st.title("DeRAG")
st.caption("Decentralized, Verifiable RAG with Trust-Weighted Ranking")

# --- Audit trail badge system -----------------------------------------
#
# The sidebar audit trail is this product's thesis made visible: whether a
# source is cryptographically verified, not just "a data field among
# others." Each badge is a real icon (not emoji) in one consistent stroke,
# with its own solid, WCAG-AA-passing color so it reads correctly
# regardless of Streamlit's light/dark theme, and it leads its row instead
# of trailing three lines of metadata. Rows also carry a state-colored left
# accent so verified/error rows are distinguishable at a glance without
# reading the badge text, and each state's icon is a distinct silhouette
# (check / triangle / circle / square) so the states don't rely on hue
# alone.

_ICONS = {
    "verified": (
        '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M4 10.5l4 4 8-9"/></svg>'
    ),
    "unverified": (
        '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M10 3l8.5 14.5H1.5L10 3z"/>'
        '<path d="M10 8.2v3.6"/><circle cx="10" cy="14.4" r="0.15" fill="currentColor"/></svg>'
    ),
    "unknown": (
        '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="8"/>'
        '<path d="M7.6 7.8a2.4 2.4 0 1 1 3.4 2.2c-.9.5-1.2.9-1.2 1.9"/>'
        '<circle cx="10" cy="14.3" r="0.15" fill="currentColor"/></svg>'
    ),
    "error": (
        '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="3" width="14" height="14" rx="2"/>'
        '<path d="M7.2 7.2l5.6 5.6M12.8 7.2l-5.6 5.6"/></svg>'
    ),
}

_STATE_LABEL = {
    "verified": "Hash match",
    "unverified": "Hash MISMATCH",
    "unknown": "Not checked",
    "error": "No registry record",
}

_STATE_DESC = {
    "verified": "content hash matches the on-chain record",
    "unverified": "content hash does not match — integrity failure",
    "unknown": "registry responded, but IPFS could not be reached to verify",
    "error": "not registered on-chain, or the registry is unreachable",
}

_BADGE_CSS = """
<style>
.derag-audit-row {
    border: 1px solid var(--secondary-background-color, rgba(128,128,128,0.25));
    border-radius: 0.5rem;
    padding: 0.75rem 0.85rem;
    margin-bottom: 0.65rem;
}
.derag-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.28rem 0.75rem 0.28rem 0.6rem;
    border-radius: 999px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.01em;
    color: #ffffff;
    line-height: 1.6;
}
.derag-badge svg { width: 15px; height: 15px; flex: none; }
.derag-badge--verified   { background: #15803d; }
.derag-badge--unverified { background: #b91c1c; }
.derag-badge--unknown    { background: #475569; }
.derag-badge--error      { background: #78716c; }
.derag-hash {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 0.55rem;
    color: var(--text-color, inherit);
    word-break: break-all;
}
.derag-audit-field {
    display: flex;
    gap: 0.4rem;
    font-size: 0.82rem;
    margin-top: 0.2rem;
    color: var(--text-color, inherit);
}
.derag-audit-field__key {
    flex: none;
    width: 5.4rem;
    opacity: 0.62;
}
.derag-audit-field__val {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums;
    word-break: break-all;
}
.derag-audit-field__val--prose {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    word-break: normal;
    overflow-wrap: break-word;
}
.derag-audit-diagnostic {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.72rem;
    opacity: 0.62;
    margin-top: 0.45rem;
    word-break: break-word;
}
.derag-legend {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    margin-bottom: 0.9rem;
}
.derag-legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.derag-legend-item .derag-badge {
    padding: 0.14rem 0.55rem 0.14rem 0.45rem;
    font-size: 0.7rem;
}
.derag-legend-item .derag-badge svg { width: 11px; height: 11px; }
.derag-legend-desc {
    font-size: 0.74rem;
    opacity: 0.72;
}
.derag-empty-state {
    font-size: 0.85rem;
    opacity: 0.72;
    margin-bottom: 0.9rem;
}
</style>
"""


def _badge_html(state: str, label: str) -> str:
    return f'<span class="derag-badge derag-badge--{state}">{_ICONS[state]}{html.escape(label)}</span>'


def _field_html(key: str, value, prose: bool = False) -> str:
    val_class = "derag-audit-field__val derag-audit-field__val--prose" if prose else "derag-audit-field__val"
    return (
        '<div class="derag-audit-field">'
        f'<span class="derag-audit-field__key">{html.escape(key)}</span>'
        f'<span class="{val_class}">{html.escape(str(value))}</span>'
        "</div>"
    )


def render_legend() -> None:
    items = "".join(
        '<div class="derag-legend-item">'
        f'{_badge_html(state, _STATE_LABEL[state])}'
        f'<span class="derag-legend-desc">{html.escape(_STATE_DESC[state])}</span>'
        "</div>"
        for state in ("verified", "unverified", "unknown", "error")
    )
    st.markdown(f'<div class="derag-legend">{items}</div>', unsafe_allow_html=True)


def render_audit_row(row: dict, ranking: dict | None = None) -> None:
    doc_hash_short = html.escape(row["doc_hash_short"])

    if "error" in row:
        label = (
            "Registry unreachable" if "unreachable" in row["error"] else _STATE_LABEL["error"]
        )
        body = (
            f'{_badge_html("error", label)}'
            f'<div class="derag-hash">{doc_hash_short}</div>'
            f'<div class="derag-audit-diagnostic">{html.escape(row["error"])}</div>'
        )
        state = "error"
    else:
        if row["verified"] is True:
            state = "verified"
        elif row["verified"] is False:
            state = "unverified"
        else:
            state = "unknown"
        badge = _badge_html(state, _STATE_LABEL[state])

        ranking_field = _field_html("Ranking", ranking["display"], prose=True) if ranking else ""
        fields = (
            ranking_field
            + _field_html("IPFS", row["ipfs_cid"])
            + _field_html("Publisher", row["publisher"])
            + _field_html("Reputation", row["reputation"])
        )
        diagnostic = (
            f'<div class="derag-audit-diagnostic">{html.escape(row["verify_error"])}</div>'
            if row["verified"] is None
            else ""
        )
        body = f'{badge}<div class="derag-hash">{doc_hash_short}</div>{fields}{diagnostic}'

    st.markdown(f'<div class="derag-audit-row derag-audit-row--{state}">{body}</div>', unsafe_allow_html=True)


def _ranking_for(doc: dict, mode: str) -> dict:
    if mode == DERAG_MODE and "score" in doc:
        display = (
            f'{doc["score"]:.3f} trust  '
            f'(similarity {doc["similarity"]:.3f}, reputation {doc["normalized_reputation"]:.3f})'
        )
    else:
        display = f'{doc["similarity"]:.3f} similarity'
    return {"display": display}


@st.cache_resource
def get_collection():
    return init_store()


query_text = st.text_input("Ask a question")
mode = st.radio("Retrieval mode", [STANDARD_RAG_MODE, DERAG_MODE])

context_docs = None

if query_text:
    try:
        collection = get_collection()
        context_docs = get_context_docs(query_text, mode, collection=collection)
    except (chromadb.errors.ChromaError, huggingface_hub.errors.HfHubHTTPError, OSError) as exc:
        st.error(f"Vector store unavailable: {exc}")
        context_docs = None
    except RegistryError as exc:
        st.error(f"Blockchain registry unavailable — is the Hardhat node running? ({exc})")
        context_docs = None

    if context_docs is not None:
        if not context_docs:
            st.warning("No matching context found for this query.")
        else:
            prompt = build_prompt(query_text, context_docs)
            try:
                st.write_stream(generate_streaming(prompt))
            except GenerationError as exc:
                st.error(f"LLM unreachable — is Ollama running? ({exc})")

with st.sidebar:
    st.markdown(_BADGE_CSS, unsafe_allow_html=True)
    st.subheader("Audit trail")
    render_legend()
    if context_docs:
        for doc in context_docs[:3]:
            doc_hash = doc["metadata"]["doc_hash"]
            render_audit_row(format_audit_row(doc_hash), ranking=_ranking_for(doc, mode))
    else:
        st.markdown(
            '<div class="derag-empty-state">Ask a question above to see the audit trail '
            "for each source, including its verification state.</div>",
            unsafe_allow_html=True,
        )
