from pipeline.ingest_prep import ingest_prep
from retrieval.standard_rag import standard_rag_answer_context
from retrieval.vector_store import add_documents, init_store

SAMPLE_TOPICS = {
    "solar.txt": "Solar panels convert sunlight directly into electricity using the photovoltaic effect.",
    "pasta.txt": "To cook pasta al dente, boil salted water and simmer the noodles for eight to ten minutes.",
    "jazz.txt": "Bebop jazz emerged in the 1940s with fast tempos and complex chord improvisation.",
    "volcano.txt": "Shield volcanoes form from slow-flowing basaltic lava that spreads over wide areas.",
}


def _seed_store(collection):
    for source_file, text in SAMPLE_TOPICS.items():
        add_documents(ingest_prep(text), source_file=source_file, collection=collection)


def test_standard_rag_returns_top_3_by_default(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    _seed_store(collection)

    hits = standard_rag_answer_context("solar power from sunlight", collection=collection)

    assert len(hits) == 3
    assert hits[0]["metadata"]["source_file"] == "solar.txt"


def test_standard_rag_respects_custom_k(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    _seed_store(collection)

    hits = standard_rag_answer_context("jazz music", k=1, collection=collection)

    assert len(hits) == 1
    assert hits[0]["metadata"]["source_file"] == "jazz.txt"
