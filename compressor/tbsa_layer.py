"""
TBSA — Template-Based Structural Abstraction
Strips grammatical noise and maps content into dense key-value schema.
Best for: meeting notes, emails, status updates.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass
class TBSAResult:
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float

def _count(text: str) -> int:
    return len(text.split())

# Filler phrases to strip
_FILLERS = re.compile(
    r'\b(the|a|an|is|are|was|were|will be|has been|have been|it|this|that|these|those|'
    r'there|here|which|who|whom|whose|what|when|where|how|in order to|so that|such that|'
    r'as well as|in addition to|furthermore|moreover|however|therefore|thus|hence|'
    r'in the context of|with respect to|with regard to|it is important to note that|'
    r'please note that|as mentioned|as stated|as noted|in summary|to summarize)\b',
    re.IGNORECASE
)

_SCHEMA_KEYS = {
    r'\b(meeting|conference|session|call)\b': 'MTG',
    r'\b(deadline|due date|due by|by)\b': 'DUE',
    r'\b(assigned to|owner|responsible|lead)\b': 'OWN',
    r'\b(status|state|current state)\b': 'STAT',
    r'\b(priority|urgent|critical|high priority)\b': 'PRI',
    r'\b(action item|task|todo|to-do|follow up)\b': 'ACT',
    r'\b(result|outcome|output|conclusion|finding)\b': 'OUT',
    r'\b(problem|issue|bug|error|defect)\b': 'ERR',
    r'\b(solution|fix|resolution|resolved)\b': 'FIX',
    r'\b(date|time|schedule|scheduled)\b': 'DT',
    r'\b(location|place|venue|room)\b': 'LOC',
    r'\b(subject|topic|regarding|re:)\b': 'SUB',
    r'\b(version|release|build)\b': 'VER',
    r'\b(requirement|spec|specification)\b': 'REQ',
    r'\b(cost|price|budget|expense)\b': 'CST',
}

def _extract_sentences(text: str) -> list[str]:
    text = re.sub(r'-{3,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    return [s.strip() for s in sentences if s.strip() and len(s.split()) > 2]

def _abstract_sentence(sentence: str) -> str:
    # Apply schema key replacements
    for pattern, key in _SCHEMA_KEYS.items():
        sentence = re.sub(pattern, key, sentence, flags=re.IGNORECASE)
    # Strip filler words
    sentence = _FILLERS.sub('', sentence)
    # Collapse multiple spaces
    sentence = re.sub(r'\s{2,}', ' ', sentence).strip()
    # Remove leading punctuation artifacts
    sentence = re.sub(r'^[,;:\s]+', '', sentence)
    return sentence.strip()

def _to_schema(sentences: list[str]) -> str:
    """Convert sentences to dense key-value schema lines."""
    schema_lines = []
    for sent in sentences:
        abstracted = _abstract_sentence(sent)
        if len(abstracted.split()) < 2:
            continue
        # If it looks like a heading, keep as-is
        if re.match(r'^#+\s', sent) or re.match(r'^\d+\.', sent):
            schema_lines.append(abstracted)
        else:
            # Wrap in minimal schema notation
            schema_lines.append(abstracted)
    return '\n'.join(schema_lines)

def compress(text: str, aggressiveness: float = 0.6) -> TBSAResult:
    """
    Apply TBSA compression.
    aggressiveness: 0.0 = light stripping, 1.0 = maximum abstraction
    """
    original_tokens = _count(text)
    sentences = _extract_sentences(text)
    
    # Keep top sentences by information density
    keep_count = max(5, int(len(sentences) * (1 - aggressiveness * 0.4)))
    
    # Score by length and keyword density
    def score(s):
        words = s.split()
        keyword_hits = sum(1 for p in _SCHEMA_KEYS for _ in re.findall(p, s, re.IGNORECASE))
        has_number = len(re.findall(r'\d+', s))
        return len(words) + keyword_hits * 2 + has_number * 3

    scored = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
    kept_indices = sorted([i for i, _ in scored[:keep_count]])
    kept_sentences = [sentences[i] for i in kept_indices]
    
    schema = _to_schema(kept_sentences)
    compressed_tokens = _count(schema)

    return TBSAResult(
        compressed_text=schema,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(original_tokens / max(compressed_tokens, 1), 2),
    )
