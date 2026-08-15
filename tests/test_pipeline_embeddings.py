# tests/test_pipeline_embeddings.py
from pipeline.embeddings import cosine_similarity, embed, _get_model


def test_embed_returns_384_dim_vectors():
    vectors = embed(["hello world", "another sentence"])
    assert len(vectors) == 2
    for v in vectors:
        assert len(v) == 384


def test_semantic_similarity_ranks_paraphrase_above_unrelated():
    anchor = "The cat sat on the mat."
    paraphrase = "A cat was sitting on the mat."
    unrelated = "The stock market fell sharply today."

    vecs = embed([anchor, paraphrase, unrelated])
    sim_paraphrase = cosine_similarity(vecs[0], vecs[1])
    sim_unrelated = cosine_similarity(vecs[0], vecs[2])

    assert sim_paraphrase > sim_unrelated


def test_model_max_seq_length_is_256():
    model = _get_model()
    assert model.max_seq_length == 256


def test_model_pooling_mode_is_mean():
    model = _get_model()
    pooling_module = model[1]
    assert pooling_module.pooling_mode == "mean"
