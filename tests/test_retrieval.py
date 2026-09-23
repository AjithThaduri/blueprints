from blueprints.retrieval import BM25, DenseIndex, Hit, reciprocal_rank_fusion, tokenize


def test_tokenize_drops_stopwords_and_keeps_codes():
    assert tokenize("What is the SKU-42 return window?") == ["sku-42", "return", "window"]


def test_bm25_prefers_rare_matching_terms():
    idx = BM25().add_many(
        [("a", "returns and refunds policy"), ("b", "warranty on batteries"), ("c", "returns returns")]
    )
    top = idx.search("battery warranty")
    assert top[0].id == "b"


def test_bm25_rejects_duplicate_ids():
    idx = BM25()
    idx.add("x", "one")
    try:
        idx.add("x", "two")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_dense_index_uses_any_embedding_function():
    vocab = ["cat", "dog", "car"]
    embed = lambda texts: [[t.count(w) for w in vocab] for t in texts]
    idx = DenseIndex(embed).add_many([("1", "cat cat"), ("2", "car"), ("3", "dog")])
    assert idx.search("car", k=1)[0].id == "2"


def test_rrf_rewards_agreement_between_rankers():
    a = [Hit("x", 9), Hit("y", 5), Hit("z", 1)]
    b = [Hit("y", 0.9), Hit("z", 0.8), Hit("x", 0.1)]
    fused = reciprocal_rank_fusion([a, b], top=3)
    assert fused[0].id == "y"
