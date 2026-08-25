from blockchain.registry_client import RegistryError, get_document
from retrieval.rerank import trust_weighted_rerank
from retrieval.standard_rag import standard_rag_answer_context
from storage.ipfs_client import IPFSError, verify_hash

STANDARD_RAG_MODE = "Standard RAG"
DERAG_MODE = "DeRAG"


def format_hash_short(h: str, n: int = 10) -> str:
    if len(h) <= n + 4:
        return h
    return f"{h[:n]}...{h[-4:]}"


def format_audit_row(doc_hash: str) -> dict:
    try:
        doc = get_document(doc_hash)
    except RegistryError as exc:
        return {
            "doc_hash_short": format_hash_short(doc_hash),
            "error": f"registry unreachable: {exc}",
        }

    row = {
        "doc_hash_short": format_hash_short(doc_hash),
        "ipfs_cid": doc["ipfs_cid"],
        "publisher": doc["publisher"],
        "reputation": doc["reputation_score"],
    }
    try:
        row["verified"] = verify_hash(doc["ipfs_cid"], doc_hash)
    except IPFSError as exc:
        row["verified"] = None
        row["verify_error"] = f"IPFS unreachable: {exc}"
    return row


def get_context_docs(query_text: str, mode: str, collection=None) -> list[dict]:
    if mode == STANDARD_RAG_MODE:
        return standard_rag_answer_context(query_text, collection=collection)
    if mode == DERAG_MODE:
        return trust_weighted_rerank(query_text)
    raise ValueError(f"unknown mode: {mode!r}")
