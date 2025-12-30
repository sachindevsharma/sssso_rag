from typing import List
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder

def rerank_with_cross_encoder(CONFIG, query: str, documents: List[Document], logger) -> List[Document]:
    """Re-rank documents using cross-encoder"""
    CROSS_ENCODER = CrossEncoder(CONFIG.RERANKER.cross_encoder.model)
    if not documents:
        return documents
    
    try:
        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get cross-encoder scores
        scores = CROSS_ENCODER.predict(pairs)
        
        # Attach scores to documents metadata and sort
        for doc, score in zip(documents, scores):
            doc.metadata['cross_encoder_score'] = float(score)
            doc.metadata['rerank_score'] = float(score)  # Keep for backward compatibility
        
        # Sort by re-ranking scores (descending)
        documents.sort(key=lambda x: x.metadata.get('cross_encoder_score', 0), reverse=True)
        
        return documents
        
    except Exception as e:
        logger.error(f" Cross-encoder re-ranking failed: {e}")
        return documents
    