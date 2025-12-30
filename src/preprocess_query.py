import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from utils import AcronymExpander, LanguageDetector


@dataclass
class QueryProcessor:
    query: str
    LOGGER: logging.Logger

    def __post_init__(self):
        self.LanDetector = LanguageDetector()
        self.AcronymExpander = AcronymExpander()
        self._detected_language_info = self.LanDetector.detect_language(self.query)


    def execute(self, session_id: Optional[str] = None) -> Tuple[str, int]:
        """Process the query by expanding acronyms and optionally rewriting it."""
        expanded_query, expansion_time_ms = self.expand_query(self.query, session_id)
        return expanded_query, expansion_time_ms

    def expand_query(self, original_query: str, session_id: Optional[str] = None) -> Tuple[str, int]:
        """First expand acronyms, then optionally rewrite using history for retrieval."""
        expansion_start = time.time()

        # 0) Detect language and check if it's a simple greeting
        self._detected_language_info = self.LanguageDetector.detect_language(original_query)
        is_greeting = self._detected_language_info.get('is_greeting', False)
        
        # If it's just a greeting without context, skip expansion entirely
        if is_greeting:
            self.LOGGER.info(f"🔍 Detected simple greeting, skipping query expansion")
            expansion_time_ms = int((time.time() - expansion_start) * 1000)
            return original_query, expansion_time_ms

        # 1) Acronym expansion (non-LLM) on the original input
        acronym_expanded = self.AcronymExpander.expand_acronyms_in_text(original_query)

        # 2) Decide if rewrite is needed using last 6 expanded queries from session history
        # Pass session_id as parameter to avoid race conditions with concurrent requests
        rewritten_or_original, _ = self._maybe_rewrite_query_with_history(session_id, acronym_expanded, original_query)

        # 3) Skip any additional LLM expansion (disabled by design)
        expansion_time_ms = int((time.time() - expansion_start) * 1000)
        return rewritten_or_original, expansion_time_ms
    
    def _maybe_rewrite_query_with_history(self, session_id: Optional[str], query: str, original_query: str = None) -> Tuple[str, bool]:
        """Use LLM to decide whether to rewrite the query using conversation history.
        Returns (final_query, rewritten_flag).
        """
        try:
            if not self.llm_client:
                return query, False
            
            # Get detected language info to preserve language
            detected_lang_info = self._detected_language_info or self.language_detector.detect_language(query)
            detected_lang = detected_lang_info.get('language', 'en')
            preserve_language = detected_lang in ['hi', 'hinglish']
            
            # Prepare last 6 expanded queries as history (most recent first)
            history_items = []
            if hasattr(self, "memory") and session_id:
                recent = self.memory.get_recent(session_id, limit=int(self.config.get("session_memory_history_len", 6)))
                # Reverse to show most recent first
                history_items = [f"{i+1}) {q}" for i, q in enumerate(reversed(recent))]
            # If no history, skip LLM call and return query as-is
            if not history_items:
                return query, False
            history_text = "\n".join(history_items)
            
            # Build language preservation instruction
            language_instruction = ""
            if preserve_language:
                if detected_lang == 'hi':
                    language_instruction = "\nCRITICAL LANGUAGE PRESERVATION RULE:\n- The user's query is in Hindi (हिंदी). You MUST preserve the Hindi language in your rewritten query.\n- Do NOT translate to English. Keep all Hindi words and phrases exactly as they are.\n- If you rewrite, the output MUST be in Hindi.\n\n"
                elif detected_lang == 'hinglish':
                    language_instruction = "\nCRITICAL LANGUAGE PRESERVATION RULE:\n- The user's query is in Hinglish (Hindi-English mix). You MUST preserve the Hinglish language style in your rewritten query.\n- Do NOT translate to pure English or pure Hindi. Keep the Hinglish mix as it is.\n- If you rewrite, the output MUST be in Hinglish.\n\n"

            # Load query rewrite prompt from file
            system_prompt = self.load_prompt("query_rewrite_prompt", "rag_prompts", self.profile_config)
            
            # Add language preservation instruction to the prompt
            if language_instruction:
                system_prompt = language_instruction + system_prompt

            user_prompt = (
                "---\n"
                f"History (newest to oldest; first is most recent):\n{history_text}\n"
                f"User: {query}"
            )
            resp = self.llm_client.chat.completions.create(
                model=self.config["llm_model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.0,
                stream=False,
            )
            content = (resp.choices[0].message.content or "").strip()
            lower = content.lower()
            if lower.startswith("rewritten:"):
                return content.split(":", 1)[1].strip(), True
            if lower.startswith("original:"):
                # If model echoed original text, we can trust it; else fall back to provided query
                text = content.split(":", 1)[1].strip()
                return (text or query), False
            # Fallback
            return query, False
        except Exception as e:
            self.LOGGER.warning(f"Rewrite-with-history failed: {e}")
            return query, False

    