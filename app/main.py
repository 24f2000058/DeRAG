import chromadb.errors
import huggingface_hub.errors
import streamlit as st

from app.ui_helpers import DERAG_MODE, STANDARD_RAG_MODE, format_audit_row, get_context_docs
from generation.llm_client import GenerationError, build_prompt, generate_streaming
from retrieval.vector_store import init_store

st.set_page_config(page_title="DeRAG", page_icon="🔗")

st.title("DeRAG")
st.caption("Decentralized, Verifiable RAG with Trust-Weighted Ranking")


@st.cache_resource
def get_collection():
    return init_store()


query_text = st.text_input("Ask a question")
mode = st.radio("Retrieval mode", [STANDARD_RAG_MODE, DERAG_MODE])

if query_text:
    try:
        collection = get_collection()
        context_docs = get_context_docs(query_text, mode, collection=collection)
    except (chromadb.errors.ChromaError, huggingface_hub.errors.HfHubHTTPError, OSError) as exc:
        st.error(f"Vector store unavailable: {exc}")
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
                st.subheader("Audit trail")
                for doc in context_docs[:3]:
                    doc_hash = doc["metadata"]["doc_hash"]
                    row = format_audit_row(doc_hash)
                    st.markdown(f"**{row['doc_hash_short']}**")
                    if "error" in row:
                        st.error(row["error"])
                    else:
                        st.write(f"IPFS CID: {row['ipfs_cid']}")
                        st.write(f"Publisher: {row['publisher']}")
                        st.write(f"Reputation: {row['reputation']}")
                        if row["verified"] is True:
                            st.success("✅ verified")
                        elif row["verified"] is False:
                            st.warning("⚠️ unverified")
                        else:
                            st.info(f"❔ verification unknown: {row.get('verify_error', '')}")
                    st.divider()
