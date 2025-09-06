"""Utility functions for RAG-Guardian."""

import hashlib
import re
from typing import List, Dict, Any
import logging

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def calculate_text_hash(text: str) -> str:
    """Calculate MD5 hash of text for caching."""
    return hashlib.md5(text.encode()).hexdigest()

def clean_code_snippet(text: str) -> str:
    """Clean and format code snippets."""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    # Remove common prefixes
    text = re.sub(r'^(```\w*\n|```\n)', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```$', '', text)
    
    return text.strip()

def extract_code_language(text: str) -> str:
    """Extract programming language from code block."""
    match = re.match(r'```(\w+)', text)
    return match.group(1) if match else "text"

def format_confidence_score(score: float) -> str:
    """Format confidence score for display."""
    percentage = int(score * 100)
    if percentage >= 90:
        return f"🟢 {percentage}% (High Confidence)"
    elif percentage >= 70:
        return f"🟡 {percentage}% (Medium Confidence)"
    else:
        return f"🔴 {percentage}% (Low Confidence)"

def parse_critique_response(critique: str) -> Dict[str, Any]:
    """Parse critique response to extract structured feedback."""
    critique_lower = critique.lower().strip()
    
    # Check if approved
    is_approved = "approved" in critique_lower or "approve" in critique_lower
    
    # Extract flaws
    flaws = []
    if "flaw:" in critique_lower:
        flaw_matches = re.findall(r'flaw:\s*([^.]+)', critique, re.IGNORECASE)
        flaws.extend(flaw_matches)
    
    # Extract suggestions
    suggestions = []
    if "suggest" in critique_lower:
        suggestion_matches = re.findall(r'suggest[^.]*:\s*([^.]+)', critique, re.IGNORECASE)
        suggestions.extend(suggestion_matches)
    
    return {
        "is_approved": is_approved,
        "flaws": flaws,
        "suggestions": suggestions,
        "raw_critique": critique
    }

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)
