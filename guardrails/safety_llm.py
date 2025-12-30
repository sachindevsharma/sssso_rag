import os
from dataclasses import dataclass
from typing import Optional, Any, Dict


def build_system_prompt() -> str:
    """Build the system prompt for safety classification"""
    try:
        if not load_prompt:
            raise ValueError("load_prompt function not available")
        if not PROFILE_CONFIG:
            raise ValueError("PROFILE_CONFIG not loaded")
        return load_prompt("safety_system_prompt", "rag_prompts", PROFILE_CONFIG)
    except Exception as e:
        logger.error(f"Failed to load safety prompt from file: {e}")
        raise

@dataclass
class SafetyLLM:
    """
    Safety checker using Meta-Llama-3.1-70B-Instruct-Turbo
    
    Response format: "safe" or "unsafe S15,S16"
    Uses the same LLM model as the RAG generation agent.
    """
    
    def __init__(self, client: Optional[Any] = None, api_key: str = None, model: str = None, base_url: str = None):
        """
        Initialize Safety LLM checker
        
        Args:
            client: Optional OpenAI client to reuse (if provided, api_key/model/base_url are ignored)
            api_key: API key (defaults to API_KEY env variable, only used if client is None)
            model: Model name (defaults to LLM_MODEL env variable, only used if client is None)
            base_url: Base URL for API (defaults to LLM_BASE_URL env variable, only used if client is None)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        
        self.system_prompt = build_system_prompt()
        
        # Use provided client or create a new one
        if client is not None:
            self.client = client
            # Get model and base_url from CONFIG (already loaded at module level)
            self.model = CONFIG.get("llm_model", model or LLM_MODEL) if CONFIG else (model or LLM_MODEL)
            self.base_url = CONFIG.get("llm_base_url", base_url or LLM_BASE_URL) if CONFIG else (base_url or LLM_BASE_URL)
            logger.info(f"Initialized Safety LLM with provided client (model: {self.model}, base_url: {self.base_url})")
        else:
            # Create new client (backward compatibility)
            self.api_key = api_key or API_KEY
            self.model = model or LLM_MODEL
            self.base_url = base_url or LLM_BASE_URL
            
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=3
                )
                logger.info(f"Initialized Safety LLM with new client (model: {self.model}, base_url: {self.base_url})")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
                raise
    
    def check(self, text: str, stage: str = "input") -> Dict[str, Any]:
        """
        Check if text is safe
        
        Args:
            text: Text to check
            stage: "User" for user input, "Agent" for model output (not used in this implementation)
            
        Returns:
            Dictionary with 'allowed', 'categories', 'explanation'
        """
        try:
            logger.info(f"[Safety LLM] Checking {stage} (length: {len(text)})")

            # Short-circuit: Political keyword filter (S15) — no LLM call if matched
            if is_political_query(text):
                logger.info("[Safety LLM] Political keywords detected — classifying as unsafe S15 without LLM call")
                raw = "unsafe\nS15"
                # Persist a minimal log file for this decision
                log_dir = "safety_prompt_logs"
                os.makedirs(log_dir, exist_ok=True)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                prompt_file = os.path.join(log_dir, f"safety_llm_{timestamp_str}.json")
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "system_prompt": self.system_prompt,
                        "user_message": text,
                        "stage": stage,
                        "model": self.model,
                        "timestamp": datetime.now().isoformat(),
                        "llm_skipped": True,
                        "skip_reason": "political_keywords",
                        "llm_raw_response": raw
                    }, f, indent=2, ensure_ascii=False)
                return parse_guard_response(raw)

            # Short-circuit: Religious keywords — block as S16
            relig_level = is_religious_query(text)
            if relig_level in {"high", "medium"}:
                logger.info(f"[Safety LLM] Religious keywords detected ({relig_level}) — classifying as unsafe S16 without LLM call")
                raw = "unsafe\nS16"
                log_dir = "safety_prompt_logs"
                os.makedirs(log_dir, exist_ok=True)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                prompt_file = os.path.join(log_dir, f"safety_llm_{timestamp_str}.json")
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "system_prompt": self.system_prompt,
                        "user_message": text,
                        "stage": stage,
                        "model": self.model,
                        "timestamp": datetime.now().isoformat(),
                        "llm_skipped": True,
                        "skip_reason": f"religious_keywords_{relig_level}",
                        "llm_raw_response": raw
                    }, f, indent=2, ensure_ascii=False)
                return parse_guard_response(raw)

            # Short-circuit: Information-system/internal terms — classify as S21
            if is_info_system_query(text):
                logger.info("[Safety LLM] Internal system/RAG/LLM keywords detected — classifying as unsafe S21 without LLM call")
                raw = "unsafe\nS21"
                log_dir = "safety_prompt_logs"
                os.makedirs(log_dir, exist_ok=True)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                prompt_file = os.path.join(log_dir, f"safety_llm_{timestamp_str}.json")
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "system_prompt": self.system_prompt,
                        "user_message": text,
                        "stage": stage,
                        "model": self.model,
                        "timestamp": datetime.now().isoformat(),
                        "llm_skipped": True,
                        "skip_reason": "info_system_keywords",
                        "llm_raw_response": raw
                    }, f, indent=2, ensure_ascii=False)
                return parse_guard_response(raw)

            # Short-circuit: Crime keywords — S1 (violent) / S2 (non-violent)
            crime_cat = detect_crime_category(text)
            if crime_cat in {"S1", "S2"}:
                logger.info(f"[Safety LLM] Crime keywords detected — classifying as unsafe {crime_cat} without LLM call")
                raw = f"unsafe\n{crime_cat}"
                log_dir = "safety_prompt_logs"
                os.makedirs(log_dir, exist_ok=True)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                prompt_file = os.path.join(log_dir, f"safety_llm_{timestamp_str}.json")
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "system_prompt": self.system_prompt,
                        "user_message": text,
                        "stage": stage,
                        "model": self.model,
                        "timestamp": datetime.now().isoformat(),
                        "llm_skipped": True,
                        "skip_reason": f"crime_keywords_{crime_cat}",
                        "llm_raw_response": raw
                    }, f, indent=2, ensure_ascii=False)
                return parse_guard_response(raw)

            # Short-circuit: Abusive/profane keywords — classify as S11 (Sexual Content/Profanity)
            if is_abusive_query(text):
                logger.info("[Safety LLM] Abusive/profane keywords detected — classifying as unsafe S11 without LLM call")
                raw = "unsafe\nS11"
                log_dir = "safety_prompt_logs"
                os.makedirs(log_dir, exist_ok=True)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                prompt_file = os.path.join(log_dir, f"safety_llm_{timestamp_str}.json")
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "system_prompt": self.system_prompt,
                        "user_message": text,
                        "stage": stage,
                        "model": self.model,
                        "timestamp": datetime.now().isoformat(),
                        "llm_skipped": True,
                        "skip_reason": "abusive_keywords",
                        "llm_raw_response": raw
                    }, f, indent=2, ensure_ascii=False)
                return parse_guard_response(raw)
            
            # Log the complete prompt for debugging
            log_data = {
                "system_prompt": self.system_prompt,
                "user_message": text,
                "stage": stage,
                "timestamp": datetime.now().isoformat(),
                "model": self.model
            }
            
            # Save prompt to file
            log_dir = "safety_prompt_logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            prompt_file = os.path.join(log_dir, f"safety_llm_{timestamp_str}.json")
            
            with open(prompt_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[Safety LLM] Saved prompt to: {prompt_file}")
            logger.info(f"[Safety LLM] User query: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            # Call LLM using OpenAI client
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stream=False
                )
                
                # Extract response content
                raw = ""
                if response.choices and len(response.choices) > 0:
                    raw = response.choices[0].message.content.strip()
                
                if not raw:
                    logger.warning(f"[Safety LLM] Empty response from API")
                    raw = ""
                
                # Save API response metadata
                log_data["api_response_metadata"] = {
                    "model": response.model if hasattr(response, 'model') else self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else None,
                        "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else None,
                        "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else None,
                    }
                }
                
            except Exception as e:
                # Check if this is a BadRequestError (Azure content filter)
                is_bad_request = BadRequestError and isinstance(e, BadRequestError)
                error_str = str(e)
                error_str_lower = error_str.lower()
                
                # Handle Azure content filter errors (BadRequestError with content_filter code)
                if is_bad_request and ('content_filter' in error_str_lower or 'content management policy' in error_str_lower):
                    logger.warning(f"[Safety LLM] Azure content filter blocked the request - treating as unsafe")
                    
                    # Determine category based on what was filtered in the error message
                    categories = []
                    
                    # Check for specific filter types in error message
                    if "'violence':" in error_str or '"violence":' in error_str:
                        if "'filtered': True" in error_str or '"filtered": true' in error_str_lower:
                            categories.append('S1')  # Violent crimes
                    if "'self_harm':" in error_str or '"self_harm":' in error_str or 'self-harm' in error_str_lower:
                        if "'filtered': True" in error_str or '"filtered": true' in error_str_lower:
                            categories.append('S10')  # Self-harm
                    if "'sexual':" in error_str or '"sexual":' in error_str:
                        if "'filtered': True" in error_str or '"filtered": true' in error_str_lower:
                            categories.append('S11')  # Sexual content
                    if "'hate':" in error_str or '"hate":' in error_str:
                        if "'filtered': True" in error_str or '"filtered": true' in error_str_lower:
                            categories.append('S9')  # Hate speech
                    
                    # Fallback: check error message keywords if no specific filter found
                    if not categories:
                        if 'violence' in error_str_lower and ('filtered' in error_str_lower or 'medium' in error_str_lower or 'high' in error_str_lower):
                            categories = ['S1']  # Violent crimes
                        elif 'self_harm' in error_str_lower or 'self-harm' in error_str_lower:
                            categories = ['S10']  # Self-harm
                        elif 'sexual' in error_str_lower:
                            categories = ['S11']  # Sexual content
                        elif 'hate' in error_str_lower:
                            categories = ['S9']  # Hate speech
                        else:
                            # Default to S1 for any content filter block
                            categories = ['S1']
                    
                    raw = f"unsafe\n{','.join(categories)}"
                    
                    # Save the Azure filter result
                    log_data["azure_content_filter"] = {
                        "blocked": True,
                        "categories": categories,
                        "error_message": error_str
                    }
                    log_data["llm_raw_response"] = raw
                    log_data["llm_skipped"] = True
                    log_data["skip_reason"] = "azure_content_filter"
                    
                    with open(prompt_file, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, indent=2, ensure_ascii=False)
                    
                    result = parse_guard_response(raw)
                    logger.info(
                        f"[Safety LLM] Azure filter result: {'SAFE' if result['allowed'] else 'UNSAFE'} "
                        f"(categories: {result['categories']})"
                    )
                    return result
                
                # Other errors (including non-content-filter BadRequestError) - log and fallback
                if is_bad_request:
                    logger.error(f"[Safety LLM] BadRequestError from LLM API: {e}", exc_info=True)
                else:
                    logger.error(f"[Safety LLM] Error calling LLM API: {e}", exc_info=True)
                
                return {
                    "allowed": True,
                    "categories": [],
                    "explanation": f"Bypassed due to API error: {str(e)}",
                    "raw_response": ""
                }
                logger.error(f"[Safety LLM] Error calling LLM API: {e}", exc_info=True)
                # Fallback: allow on error
                return {
                    "allowed": True,
                    "categories": [],
                    "explanation": f"Bypassed due to API error: {str(e)}",
                    "raw_response": ""
                }
            
            # Save response to same file
            log_data["llm_raw_response"] = raw
            log_data["llm_response_length"] = len(raw)
            
            with open(prompt_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[Safety LLM] Raw response (length {len(raw)}): {raw!r}")
            
            # Parse the response
            result = parse_guard_response(raw)
            
            # Save parsed result
            log_data["parsed_result"] = result
            
            with open(prompt_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            logger.info(
                f"[Safety LLM] Result: {'SAFE' if result['allowed'] else 'UNSAFE'} "
                f"(categories: {result['categories']})"
            )
            logger.info(f"[Safety LLM] Complete log saved to: {prompt_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"[Safety LLM] Error during check: {e}", exc_info=True)
            # Fallback: allow on error
            return {
                "allowed": True,
                "categories": [],
                "explanation": f"Bypassed due to error: {str(e)}",
                "raw_response": ""
            }

