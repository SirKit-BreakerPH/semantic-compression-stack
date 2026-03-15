"""
DLM — Dynamic Lexicon Mapping
Replaces high-frequency multi-token phrases with short symbols + dictionary header.
Best for: long research papers, technical docs with repeated terminology.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from collections import Counter

@dataclass
class DLMResult:
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    lexicon: dict[str, str]  # symbol -> original phrase
    replacements_made: int

def _count(text: str) -> int:
    return len(text.split())

def _extract_phrases(text: str, min_words: int = 3, min_freq: int = 2) -> list[tuple[str, int]]:
    """Find repeated multi-word phrases."""
    # Clean text for phrase extraction
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    words = clean.split()
    
    phrase_counts: Counter = Counter()
    
    # Extract 3, 4, and 5-word phrases
    for n in range(min_words, 6):
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            # Skip phrases that are mostly stop words
            content_words = [w for w in phrase.split() 
                           if w not in {'the','a','an','is','are','was','were',
                                       'to','of','in','for','on','with','at','by',
                                       'from','that','this','it','as','or','and'}]
            if len(content_words) >= 2:
                phrase_counts[phrase] += 1
    
    # Return phrases that appear multiple times, sorted by length * frequency
    qualified = [(p, c) for p, c in phrase_counts.items() if c >= min_freq]
    qualified.sort(key=lambda x: len(x[0].split()) * x[1], reverse=True)
    return qualified

def _build_lexicon(phrases: list[tuple[str, int]], max_entries: int = 20) -> dict[str, str]:
    """Assign symbols to top phrases. symbol -> original phrase."""
    lexicon = {}
    for i, (phrase, _) in enumerate(phrases[:max_entries]):
        symbol = f"@{i+1}"
        lexicon[symbol] = phrase
    return lexicon

def _apply_substitution(text: str, lexicon: dict[str, str]) -> tuple[str, int]:
    """Replace phrases in text with symbols. Returns (new_text, replacements_count)."""
    result = text
    total_replacements = 0
    
    # Sort by phrase length descending to replace longest first
    sorted_lexicon = sorted(lexicon.items(), key=lambda x: len(x[1]), reverse=True)
    
    for symbol, phrase in sorted_lexicon:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        count = len(pattern.findall(result))
        if count > 0:
            result = pattern.sub(symbol, result)
            total_replacements += count
    
    return result, total_replacements

def _build_header(lexicon: dict[str, str]) -> str:
    """Build the dictionary header prepended to the compressed text."""
    if not lexicon:
        return ""
    entries = "; ".join(f"{sym}={phrase}" for sym, phrase in lexicon.items())
    return f"[LIB: {entries}]\n\n"

def compress(text: str, min_freq: int = 2, max_symbols: int = 20) -> DLMResult:
    """
    Apply DLM compression.
    min_freq: minimum times a phrase must appear to get a symbol
    max_symbols: maximum number of symbols in the lexicon
    """
    original_tokens = _count(text)
    
    # Extract and rank repeated phrases
    phrases = _extract_phrases(text, min_freq=min_freq)
    
    if not phrases:
        # No repeated phrases found — return original
        return DLMResult(
            compressed_text=text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=1.0,
            lexicon={},
            replacements_made=0,
        )
    
    # Build lexicon and apply substitutions
    lexicon = _build_lexicon(phrases, max_entries=max_symbols)
    compressed_body, replacements = _apply_substitution(text, lexicon)
    
    # Only keep lexicon entries that were actually used
    used_lexicon = {sym: phrase for sym, phrase in lexicon.items() 
                   if sym in compressed_body}
    
    if not used_lexicon:
        return DLMResult(
            compressed_text=text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=1.0,
            lexicon={},
            replacements_made=0,
        )
    
    header = _build_header(used_lexicon)
    final_text = header + compressed_body
    compressed_tokens = _count(final_text)

    return DLMResult(
        compressed_text=final_text,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(original_tokens / max(compressed_tokens, 1), 2),
        lexicon=used_lexicon,
        replacements_made=replacements,
    )

def decompress(compressed: str) -> str:
    """Reverse DLM by expanding symbols back to original phrases."""
    if not compressed.startswith('[LIB:'):
        return compressed
    
    header_end = compressed.find(']\n\n')
    if header_end == -1:
        return compressed
    
    header = compressed[5:header_end]
    body   = compressed[header_end+3:]
    
    # Parse lexicon from header
    lexicon = {}
    for entry in header.split(';'):
        entry = entry.strip()
        if '=' in entry:
            sym, phrase = entry.split('=', 1)
            lexicon[sym.strip()] = phrase.strip()
    
    # Expand symbols
    result = body
    for sym, phrase in lexicon.items():
        result = result.replace(sym, phrase)
    
    return result
