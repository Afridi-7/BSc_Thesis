from src.rag.retriever import ClinicalRetriever


def _retriever_stub() -> ClinicalRetriever:
    retriever = ClinicalRetriever.__new__(ClinicalRetriever)
    retriever.internet_enabled = True
    retriever.internet_top_k = 3
    retriever.min_chunk_length = 80
    retriever._web_cache = {}
    return retriever


def test_handle_missing_pdfs_allows_internet_mode():
    retriever = _retriever_stub()
    retriever.pdf_missing_strategy = "fail"
    retriever.pdf_download_base_url = ""
    retriever.pdf_directory = "db_/pdfs"

    retriever._handle_missing_pdfs({"missing_files": ["missing.pdf"]})


def test_merge_retrieval_results_deduplicates_and_limits():
    retriever = _retriever_stub()

    local_results = [
        {"source": "pdf_a", "chunk_id": 0, "text": "alpha", "score": 0.92},
        {"source": "pdf_b", "chunk_id": 1, "text": "beta", "score": 0.75},
    ]
    web_results = [
        {"source": "web:example.com", "chunk_id": 1, "text": "web beta", "score": 0.81},
        {"source": "web:example.com", "chunk_id": 2, "text": "web gamma", "score": 0.70},
    ]

    merged = retriever._merge_retrieval_results(local_results, web_results, 2)

    assert len(merged) == 2
    assert merged[0]["score"] >= merged[1]["score"]


def test_normalize_search_result_url_extracts_redirect_target():
    retriever = _retriever_stub()

    url = retriever._normalize_search_result_url(
        "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle"
    )

    assert url == "https://example.com/article"
