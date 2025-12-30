"""
Acronym expander module for DGCA RAG Production System
Expands acronyms in text using the acronym glossary
"""

import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class AcronymExpander:
    """
    Expands acronyms in text using the acronym glossary
    """
    
    def __init__(self, glossary_file_path: str):
        """
        Initialize the acronym expander with glossary file
        
        Args:
            glossary_file_path: Path to the acronym_glossary.csv file
        """
        self.glossary_file_path = Path(glossary_file_path)
        self.acronym_dict = self._load_acronym_glossary()
    
    def _load_acronym_glossary(self) -> Dict[str, str]:
        """
        Load acronym glossary from CSV file
        
        Returns:
            Dictionary mapping acronym to definition
        """
        acronym_dict = {}
        
        try:
            if not self.glossary_file_path.exists():
                logger.warning(f"Acronym glossary file not found: {self.glossary_file_path}")
                return acronym_dict
            
            with open(self.glossary_file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    acronym = row['Acronym'].strip()
                    definition = row['Definition'].strip()
                    if acronym and definition:
                        acronym_dict[acronym] = definition
            
            logger.info(f"Loaded {len(acronym_dict)} acronyms from glossary")
            
        except Exception as e:
            logger.error(f"Failed to load acronym glossary: {e}")
        
        return acronym_dict
    
    def expand_acronyms_in_text(self, text: str) -> str:
        """
        Expand acronyms in text using exact word boundary matching
        
        Args:
            text: Input text to expand acronyms in
            
        Returns:
            Text with acronyms expanded
        """
        if not self.acronym_dict:
            return text
        
        expanded_text = text
        
        # Sort acronyms by length (longest first) to handle overlapping cases
        sorted_acronyms = sorted(self.acronym_dict.items(), key=lambda x: len(x[0]), reverse=True)
        
        for acronym, definition in sorted_acronyms:
            # Use word boundary regex to match exact acronym
            # This ensures "AD" matches but not "ADX" or "BAD"
            pattern = r'\b' + re.escape(acronym) + r'\b'
            
            # Replace with acronym(definition) format
            replacement = f"{acronym}({definition})"
            
            # Count occurrences before replacement
            matches = len(re.findall(pattern, expanded_text))
            
            if matches > 0:
                expanded_text = re.sub(pattern, replacement, expanded_text)
                logger.debug(f"Expanded {matches} occurrences of '{acronym}' to '{replacement}'")
        
        return expanded_text
    
    def get_acronyms_in_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Get list of acronyms found in text with their definitions
        
        Args:
            text: Input text to search for acronyms
            
        Returns:
            List of tuples (acronym, definition) found in text
        """
        found_acronyms = []
        
        if not self.acronym_dict:
            return found_acronyms
        
        for acronym, definition in self.acronym_dict.items():
            # Use word boundary regex to match exact acronym
            pattern = r'\b' + re.escape(acronym) + r'\b'
            
            if re.search(pattern, text):
                found_acronyms.append((acronym, definition))
        
        return found_acronyms
    
    def expand_acronyms_in_chunks(self, chunks: List) -> List:
        """
        Expand acronyms in a list of chunks
        
        Args:
            chunks: List of Document chunks
            
        Returns:
            List of chunks with expanded acronyms
        """
        if not self.acronym_dict:
            logger.warning("No acronym dictionary loaded, skipping expansion")
            return chunks
        
        expanded_chunks = []
        total_expansions = 0
        
        for chunk in chunks:
            # Expand acronyms in chunk content
            expanded_content = self.expand_acronyms_in_text(chunk.page_content)
            
            # Count expansions made
            expansions_made = chunk.page_content != expanded_content
            if expansions_made:
                total_expansions += 1
            
            # Create new chunk with expanded content
            expanded_chunk = type(chunk)(
                page_content=expanded_content,
                metadata=chunk.metadata.copy()
            )
            
            # Add acronym expansion metadata
            from datetime import datetime
            expanded_chunk.metadata.update({
                "acronyms_expanded": expansions_made,
                "acronym_expansion_timestamp": datetime.now().isoformat()
            })
            
            expanded_chunks.append(expanded_chunk)
        
        logger.info(f"Expanded acronyms in {total_expansions}/{len(chunks)} chunks")
        return expanded_chunks
    
    def get_glossary_summary(self) -> Dict[str, any]:
        """
        Get summary of loaded acronym glossary
        
        Returns:
            Summary dictionary with glossary statistics
        """
        return {
            "total_acronyms": len(self.acronym_dict),
            "acronyms": list(self.acronym_dict.keys()),
            "glossary_file": str(self.glossary_file_path)
        }