"""Language Detection Module for DGCA RAG System
================================================

This module provides language detection capabilities to identify
Hindi, Hinglish, and English text for proper LLM response generation.

Author: DGCA RAG Team
Version: 1.0
"""

import re
import logging
from typing import Dict, Optional, Tuple, Any

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available, using fallback language detection")

# Hindi Unicode ranges
HINDI_UNICODE_RANGES = [
    (0x0900, 0x097F),  # Devanagari
    (0x1CD0, 0x1CF9),  # Vedic Extensions
    (0xA8E0, 0xA8FF),  # Devanagari Extended
]

# Common Hindi words for better detection
COMMON_HINDI_WORDS = {
    'क्या', 'कैसे', 'कहाँ', 'कब', 'कौन', 'क्यों', 'किस', 'किसी',
    'मैं', 'तुम', 'आप', 'वह', 'यह', 'हम', 'वे', 'ये',
    'है', 'हैं', 'था', 'थे', 'थी', 'हो', 'होगा', 'होगी',
    'के', 'का', 'की', 'को', 'से', 'में', 'पर', 'तक',
    'और', 'या', 'लेकिन', 'क्योंकि', 'जब', 'तब', 'अगर', 'तो'
}

# Common greetings (case-insensitive)
GREETINGS = {
    'hi', 'hello', 'hey', 'namaste', 'namaskar', 'good morning',
    'good afternoon', 'good evening', 'good night', 'greetings',
    'hi there', 'hey there', 'hello there'
}

# Hindi greetings
HINDI_GREETINGS = {
    'नमस्ते', 'नमस्कार', 'प्रणाम', 'सत श्री अकाल',
    'आदाब', 'अस्सलामु अलैकुम', 'जय श्री कृष्ण'
}


class LanguageDetector:
    """Language detector for Hindi, Hinglish, and English text"""
    
    def __init__(self):
        self.use_langdetect = LANGDETECT_AVAILABLE
        if not self.use_langdetect:
            logging.warning("Using fallback language detection (Unicode-based)")
    
    def _contains_hindi_unicode(self, text: str) -> bool:
        """Check if text contains Hindi Unicode characters"""
        for char in text:
            code_point = ord(char)
            for start, end in HINDI_UNICODE_RANGES:
                if start <= code_point <= end:
                    return True
        return False
    
    def _count_hindi_words(self, text: str) -> int:
        """Count Hindi words in text"""
        words = re.findall(r'\b\w+\b', text)
        hindi_count = 0
        for word in words:
            if word in COMMON_HINDI_WORDS:
                hindi_count += 1
        return hindi_count
    
    def _is_simple_greeting(self, text: str) -> bool:
        """Check if text is a simple greeting without context"""
        text_lower = text.strip().lower()
        
        # Remove punctuation and extra spaces
        text_clean = re.sub(r'[^\w\s]', '', text_lower).strip()
        
        # Check English greetings
        if text_clean in GREETINGS:
            return True
        
        # Check Hindi greetings
        if text_clean in HINDI_GREETINGS:
            return True
        
        # Check if it's just "hi" or "hello" with minimal context
        if len(text_clean.split()) <= 2:
            if any(greeting in text_clean for greeting in ['hi', 'hello', 'hey', 'namaste', 'namaskar']):
                return True
        
        return False
    
    def _detect_with_langdetect(self, text: str) -> Optional[str]:
        """Detect language using langdetect library"""
        try:
            # Detect primary language
            detected_lang = detect(text)
            
            # Get all language probabilities
            lang_probs = detect_langs(text)
            
            # Check if Hindi is detected with reasonable confidence
            for lang_prob in lang_probs:
                if lang_prob.lang == 'hi' and lang_prob.prob > 0.3:
                    return 'hi'
            
            # If primary is Hindi, return it
            if detected_lang == 'hi':
                return 'hi'
            
            # Check for Hinglish (mix of English and Hindi)
            if detected_lang == 'en':
                # Check if text contains Hindi characters
                if self._contains_hindi_unicode(text):
                    return 'hinglish'
                return 'en'
            
            return detected_lang
            
        except LangDetectException:
            return None
        except Exception as e:
            logging.warning(f"langdetect error: {e}")
            return None
    
    def _detect_with_fallback(self, text: str) -> str:
        """Fallback language detection using Unicode and heuristics"""
        # Check for Hindi Unicode characters
        has_hindi_unicode = self._contains_hindi_unicode(text)
        
        # Count Hindi words
        hindi_word_count = self._count_hindi_words(text)
        
        # Count total words
        total_words = len(re.findall(r'\b\w+\b', text))
        
        # Determine language
        if has_hindi_unicode:
            if total_words > 0 and hindi_word_count / total_words > 0.3:
                return 'hi'  # Mostly Hindi
            elif total_words > 0:
                return 'hinglish'  # Mix of Hindi and English
            else:
                return 'hi'  # Default to Hindi if Unicode present
        else:
            return 'en'  # Default to English
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of input text
        
        Args:
            text: Input text to detect language for
            
        Returns:
            Dict with keys:
                - language: 'hi', 'hinglish', or 'en'
                - confidence: float (0.0 to 1.0)
                - is_greeting: bool (True if text is just a greeting)
                - method: str ('langdetect' or 'fallback')
        """
        if not text or not text.strip():
            return {
                'language': 'en',
                'confidence': 0.0,
                'is_greeting': False,
                'method': 'empty'
            }
        
        text = text.strip()
        
        # Check if it's a simple greeting
        is_greeting = self._is_simple_greeting(text)
        
        # Detect language
        if self.use_langdetect:
            detected_lang = self._detect_with_langdetect(text)
            method = 'langdetect'
            
            if detected_lang is None:
                # Fallback if langdetect fails
                detected_lang = self._detect_with_fallback(text)
                method = 'fallback'
            else:
                # Check for Hinglish if langdetect says English but has Hindi chars
                if detected_lang == 'en' and self._contains_hindi_unicode(text):
                    detected_lang = 'hinglish'
        else:
            detected_lang = self._detect_with_fallback(text)
            method = 'fallback'
        
        # Calculate confidence (simple heuristic)
        if detected_lang == 'hi':
            confidence = 0.9 if self._contains_hindi_unicode(text) else 0.6
        elif detected_lang == 'hinglish':
            confidence = 0.7
        else:
            confidence = 0.8
        
        return {
            'language': detected_lang or 'en',
            'confidence': confidence,
            'is_greeting': is_greeting,
            'method': method
        }
    
    def should_preserve_language(self, text: str) -> bool:
        """
        Determine if language should be preserved (Hindi/Hinglish detected)
        
        Args:
            text: Input text
            
        Returns:
            True if language should be preserved (Hindi/Hinglish), False otherwise
        """
        result = self.detect_language(text)
        return result['language'] in ['hi', 'hinglish']
    
    def get_response_language_instruction(self, text: str) -> str:
        """
        Get language instruction for LLM prompt based on detected language
        
        Args:
            text: Input text
            
        Returns:
            Language instruction string for LLM prompt
        """
        result = self.detect_language(text)
        lang = result['language']
        
        if lang == 'hi':
            return "CRITICAL: The user's query is in Hindi. You MUST respond in Hindi (हिंदी). All sections (Summary, Elaboration, Points to Remember) must be in Hindi."
        elif lang == 'hinglish':
            return "CRITICAL: The user's query is in Hinglish (Hindi-English mix). You MUST respond in Hinglish or Hindi, matching the user's language style. All sections (Summary, Elaboration, Points to Remember) must be in Hinglish/Hindi."
        else:
            return "Respond in English."


# Global instance
_language_detector = None

def get_language_detector() -> LanguageDetector:
    """Get or create global language detector instance"""
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetector()
    return _language_detector

