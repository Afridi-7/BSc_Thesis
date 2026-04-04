"""Retrieval-Augmented Generation for clinical reasoning."""

__all__ = ["ClinicalRetriever", "ClinicalReasoner"]


def __getattr__(name):
    if name == "ClinicalRetriever":
        from src.rag.retriever import ClinicalRetriever
        return ClinicalRetriever
    if name == "ClinicalReasoner":
        from src.rag.llm_reasoner import ClinicalReasoner
        return ClinicalReasoner
    raise AttributeError(name)
