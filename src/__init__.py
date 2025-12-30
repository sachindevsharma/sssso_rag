import os
import re
import logging
from typing import List, Dict, Any

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter 

from src.retriever.extract_metadata import ExtractMetadatafromQuery
from src.reranker import Reranker

@dataclass
class pRAGma:
    CONFIG: any
    logger: logging.Logger = logging.getLogger(__name__)

    def __post_init__(self):
        # self._setup_query_logging()
        # self._setup_rerank_prompts_dir()
        pass
    
    def invoke(self, query: str):
        filters = ExtractMetadatafromQuery(query, self.CONFIG).execute()
        print(filters)
        self.logger.info(f"Metadata filters extracted: {filters}")
        ret_docs = self.retrieve_documents(query, filters, n_docs=10)

        # RR = Reranker("cross_encoder", self.CONFIG, logging.getLogger())
        # rerank_chunks = RR.execute(query, context)
        
        prompt = self.format_prompt(query, ret_docs)
        res = self.CONFIG.LLM_MODEL.invoke(prompt)
        
        return res.content

    def retrieve_documents(self, query, filters={}, n_docs=5) -> List[Document]:
        ret_docs = self.VECTOR_STORE.similarity_search_with_score(
            query, filter=filters, k=n_docs, fetch_k=100
        )
        print(f"Retrieved {len(ret_docs)} documents from vector store.")
        docs = [i for i, _ in ret_docs]

        return docs

    def format_prompt(self, question, ret_documents) -> str:
        context = self._format_context_from_docs(ret_documents)
        prompt = f"""\
Answer the user's question based ONLY on the following context. 
If the context doesn't contain the answer, say "I don't have enough information in the provided documents."

Context:
{context}

Question: 
{question}

Instructions:
- You should never mention that the context was provided to you. 
- Your tone should act like you are expert and are answering questions yourself.
    """
        return prompt

    def _format_context_from_docs(self, documents: List[Document]) -> str:
        context = []
        for doc in documents:
            section_name = doc.metadata.get("section_name", None)
            topic_num = doc.metadata.get("topic_num")
            file_name = doc.metadata.get("file_name")
        
            chunk_txt = f"The text below between <p></p> tags comes from the document titled '{file_name}' which is quoted as TOPIC {topic_num} "
        
            if section_name is not None:
                chunk_txt += f"pertaining to {section_name.upper()} section"
            
            chunk_txt += "\n<p>" + doc.page_content + "\n</p>\n"

            context.append(chunk_txt)
        return "\n\n".join(context)
        
    def load_and_ingest(self):
        documents = self.load_data()
        documents = self.update_meatdata_by_name_topic_num(documents)
        chunked_docs, unique_map = self.chunk_doc_by_headers(documents)

        vector_store = FAISS.from_documents(chunked_docs, self.CONFIG.EMBEDDINGS_MODEL)
        vector_store.save_local(self.CONFIG.PATH.vector_db)
        print(f"Unique metadata values: {unique_map}")


    def _setup_query_logging(self):
        """Setup query logging directory"""
        if self.CONFIG.save_queries:
            os.makedirs(self.CONFIG.PATH.query_logs_dir, exist_ok=True)
            self.logger.info(f" Query logs directory: {self.CONFIG.PATH.query_logs_dir}")

    def _setup_rerank_prompts_dir(self):
        """Setup LLM reranking prompts directory"""
        os.makedirs(self.config['llm_rerank_prompts_dir'], exist_ok=True)
        self.logger.info(f" LLM reranking prompts directory: {self.config['llm_rerank_prompts_dir']}")
    
    @property
    def VECTOR_STORE(self):
        v_store = FAISS.load_local(
            self.CONFIG.PATH.vector_db,  
            self.CONFIG.EMBEDDINGS_MODEL, 
            allow_dangerous_deserialization=True
        )
        return v_store



    def load_data(self):

        loader = PyPDFDirectoryLoader(
            self.CONFIG.PATH.data, 
            mode="single", 
            extract_images=True, 
            extraction_mode="plain"
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents
    
    def update_meatdata_by_name_topic_num(self, documents: List[Document]):
        for doc in documents:
            topic_name = doc.metadata.get("creator", "").split("-")
            topic_name = [i.strip() for i in topic_name]
            
            meta = doc.metadata
            meta["file_name"] = topic_name[-1]
            meta["topic_num"] = topic_name[1].replace("Topic", "").strip()

            doc.metadata = meta
        return documents
    
    def get_section_num_name(self, match_list):
        for mat in match_list:
            if mat:
                num, name = mat.groups()
                num = re.sub(r'[(){}<>.\s]', '', num)
                name = name.strip().lower()
                return num, name
        return None, None
        
    def chunk_doc_by_headers(self, documents: List[Document]):
        chunks = []
        metadata_unique_val = {}
        
        separators = [
            r"^\s*(\(\d+\)|\d+(?:(?:\.\d+)+|\.))\s+(.*)",      # Look for line starting by digits (e.g. "(6),(6.) 6.")
            r"^\s*(\([a-zA-Z]+\)|[a-zA-Z]\.)\s+(.*)"           # Look for line starting by characters (a) or (A) or (A.) or  A.
        ]

        for doc in documents:
            metadata = doc.metadata
            chunk_text = ""
            for line in doc.page_content.splitlines():
                match = [re.match(i, line) for i in separators]
                if any(match):
                    if chunk_text != "":
                        chunks.append(Document(metadata=metadata, page_content=chunk_text))
                        chunk_text = line

                    num, name = self.get_section_num_name(match)
                    if num is not None:
                        metadata["section_num"] = num
                        metadata_unique_val = self.__update_unique_map_val(metadata_unique_val, "section_num", num)
                    if name is not None:
                        metadata["section_name"] = name
                        metadata_unique_val = self.__update_unique_map_val(metadata_unique_val, "section_name", name)
                else:
                    chunk_text += "\n" + line   # adding \n as splitline was done
                    
            chunks.append(Document(metadata=metadata, page_content=chunk_text))
            
        print(f"Created {len(chunks)} chunks")
        return chunks, metadata_unique_val


    def __update_unique_map_val(self, mapping, key, val):
        if key in mapping:
            if val not in mapping[key]:
                mapping[key].append(val)
        else:
            mapping[key] = [val]
        return mapping
        
    def chunk_documents(self, documents):
        # 1. Regex Splitter: Looks for patterns like "6." or "(a)" at the start of lines.
        # (?=...) is a lookahead assertion; it splits *before* the match, keeping the header in the chunk.
        # Pattern explanation:
        # \n          -> Start with a newline
        # (?:\d+\.|   -> Match "6." (numbered headers)
        # \([a-z]\))  -> OR Match "(a)" (lettered sub-headers)
        
        separators = [
                r"\n(?=\d+\.)",      # Look for newline followed by digits and a dot (e.g. "6.")
                r"\n(?=\([a-z]\)\s)" # Look for newline followed by (a) or (b)
            ]
        
        section_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, # Larger size to fit the whole quote+insight
            chunk_overlap=0, # No overlap needed if we split perfectly by section
            separators=separators, 
            is_separator_regex=True,
            keep_separator=True
        )
        
        structured_chunks = []
        
        # Process each page
        for doc in documents:
            # Split this page into logical sections
            page_sections = section_splitter.split_documents([doc])
            structured_chunks.extend(page_sections)

        return structured_chunks

    