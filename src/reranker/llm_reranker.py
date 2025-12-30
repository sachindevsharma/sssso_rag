from typing import List
from langchain_core.documents import Document


def rerank_with_llm(
    CONFIG: any, 
    query: str, 
    documents: List[Document], 
    logger: any,
    max_len_docs: int = 1000
) -> List[Document]:
    """Re-rank documents using LLM-based relevance scoring"""
    if not documents:
        return documents
    
    try:
        logger.info(f"🤖 Starting LLM re-ranking for {len(documents)} documents")
        
        # Prepare documents for LLM evaluation with metadata
        doc_texts = []
        for i, doc in enumerate(documents):
            # Truncate content to avoid token limits
            content = doc.page_content[:max_len_docs] + "..." if len(doc.page_content) > max_len_docs else doc.page_content
            
            # Extract key metadata fields
            metadata = doc.metadata
            doc_type = metadata.get('Type', metadata.get('document_type', 'Unknown'))
            file_name = metadata.get('file_name', metadata.get('original_file_name', metadata.get('source', 'Unknown')))
            page = metadata.get('page', metadata.get('page_label', 'N/A'))
            
            # Extract Excel metadata if available
            excel_meta = metadata.get('excel_metadata', {})
            if excel_meta:
                doc_type = excel_meta.get('Type', excel_meta.get('type', doc_type))
                section = excel_meta.get('Section', excel_meta.get('section', ''))
                description = excel_meta.get('Description', excel_meta.get('description', ''))
            else:
                section = metadata.get('section', metadata.get('section_number', ''))
                description = metadata.get('description', '')
            
            # Build document entry with metadata
            doc_entry = f"Document {i+1}:\n"
            doc_entry += f"Document Type: {doc_type}\n"
            doc_entry += f"Source: {file_name}"
            if page != 'N/A':
                doc_entry += f" (Page {page})"
            doc_entry += "\n"
            # if section:
                # doc_entry += f"Section: {section}\n"
            # if description:
            #     doc_entry += f"Description: {description}\n"
            doc_entry += f"Content: {content}"
            
            doc_texts.append(doc_entry)
        
        # Create LLM re-ranking prompt
        # Load rerank prompt from file
        rerank_prompt_template = self.load_prompt("rerank_prompt", "rag_prompts", self.profile_config)
        rerank_prompt = rerank_prompt_template.format(
            query=query,
            documents=chr(10).join(doc_texts)
        )
#             rerank_prompt = f"""You are an expert DGCA (Directorate General of Civil Aviation) document relevance evaluator. 

# Your task is to score the relevance of each document to the given query on a scale of 0.0 to 1.0, where:
# - 1.0 = Highly relevant, directly answers the query
# - 0.8 = Very relevant, contains important related information
# - 0.6 = Moderately relevant, contains some useful information
# - 0.4 = Somewhat relevant, tangentially related
# - 0.2 = Minimally relevant, barely related
# - 0.0 = Not relevant at all

# QUERY: {query}

# DOCUMENTS:
# {chr(10).join(doc_texts)}

# Provide ONLY a JSON array of relevance scores in the same order as the documents, like: [0.9, 0.7, 0.3, 0.8]
# Do not include any explanations or additional text."""
        # Save the reranking prompt to directory
        self._save_rerank_prompt(query, rerank_prompt, len(documents))
        
        response = self.llm_client.chat.completions.create(
            model=self.config["llm_model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a DGCA document relevance evaluator. Return only a JSON array of relevance scores."
                },
                {
                    "role": "user",
                    "content": rerank_prompt
                }
            ],
            max_tokens=200,
            temperature=0.0,
            stream=False
        )
        
        llm_response = response.choices[0].message.content.strip()
        
        # Parse LLM scores
        try:
            import json
            scores = json.loads(llm_response)
            print(scores)
            if len(scores) != len(documents):
                logger.warning(f"⚠️ LLM returned {len(scores)} scores for {len(documents)} documents, using fallback")
                return documents
            
            # Attach LLM scores to documents
            for doc, score in zip(documents, scores):
                doc.metadata['llm_rerank_score'] = float(score)
                # Update main rerank_score if no cross-encoder score exists
                if 'cross_encoder_score' not in doc.metadata:
                    doc.metadata['rerank_score'] = float(score)
            
            # Sort by LLM scores (descending)
            documents.sort(key=lambda x: x.metadata.get('llm_rerank_score', 0), reverse=True)
            
            logger.info(f"✅ LLM re-ranking completed with scores: {[round(s, 2) for s in scores]}")
            return documents
            
        except (json.JSONDecodeError, ValueError) as parse_error:
            logger.error(f"❌ Failed to parse LLM re-ranking scores: {parse_error}")
            logger.error(f"LLM response was: {llm_response}")
            return documents
            
    except Exception as e:
        logger.error(f"❌ LLM re-ranking failed: {e}")
        return documents
