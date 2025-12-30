import logging
from dataclasses import dataclass

from .cross_encoder_reranker import rerank_with_cross_encoder
from .llm_reranker import rerank_with_llm


@dataclass
class Reranker:
    method: str
    CONFIG: any
    logger: logging.Logger

    def execute(self, query, documents):
        """Rerank documents using cross-encoder and LLM"""

        encoder_map = {
            "cross_encoder": rerank_with_cross_encoder,
            "llm": rerank_with_llm
        }

        METHOD = encoder_map.get(self.method.lower(), None)
        if METHOD is None:
            raise ValueError(f"⚠️ Reranker method '{self.method}' not recognized. Skipping reranking.")
        
        # First apply cross-encoder reranking
        documents = METHOD(self.CONFIG, query, documents, self.logger)

        return documents