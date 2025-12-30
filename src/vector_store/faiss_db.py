

class FaissVectorStore:

    def retrieve_documents(self, query: str, filters: Dict[str, Any], n_docs: int):
        ret_docs = self.VECTOR_STORE.similarity_search_with_score(
            query, filter=filters, k=n_docs, fetch_k=100
        )
        print(f"Retrieved {len(ret_docs)} documents from vector store.")
        metadata = [i.metadata for i, _ in ret_docs]
        docs = [i for i, _ in ret_docs]

        return docs, metadata

    def VECTOR_STORE(self):
        if not self._vector_store:
            self._vector_store = FAISS.load_local(
                self.index_name,
                self.embeddings_model,
                faiss_index_path=self.faiss_index_path,
            )
        return self._vector_store
    
    def save_vector_store(self):
        if self._vector_store:
            self._vector_store.save_local(
                self.index_name,
                faiss_index_path=self.faiss_index_path,
            )
            self.logger.info(f"Vector store saved at {self.index_name} and {self.faiss_index_path}")

    def add_documents(self, documents: List[Document]):
        if not self._vector_store:
            self.VECTOR_STORE  # Initialize the vector store if not already done
        self._vector_store.add_documents(documents)
        self.logger.info(f"Added {len(documents)} documents to the vector store.")  

    def delete_documents_by_metadata(self, metadata_filter: Dict[str, Any]):
        if not self._vector_store:
            self.VECTOR_STORE  # Initialize the vector store if not already done
        initial_count = self._vector_store.index.ntotal
        self._vector_store.delete_documents_by_metadata(metadata_filter)
        final_count = self._vector_store.index.ntotal
        deleted_count = initial_count - final_count
        self.logger.info(f"Deleted {deleted_count} documents from the vector store based on metadata filter: {metadata_filter}")
