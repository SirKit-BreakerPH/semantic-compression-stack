"""
Layer 1 — Structural Skeletonization (PromptPacker-style)
----------------------------------------------------------
Detects content type (code vs prose) and applies the appropriate
skeleton strategy.

- Code  → tree-sitter AST: keeps imports, signatures, docstrings, types
- Prose → sentence-level extraction: keeps first + key sentences per paragraph
"""

from __future__ import annotations
import re
import textwrap
from dataclasses import dataclass
from typing import Optional

# ── tree-sitter (optional; graceful fallback if not compiled) ──────────────
try:
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjs
    from tree_sitter import Language, Parser

    PY_LANGUAGE = Language(tspython.language())
    JS_LANGUAGE = Language(tsjs.language())
    _TS_AVAILABLE = True
except Exception:
    _TS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
@dataclass
class SkeletonResult:
    skeleton: str
    original_tokens: int
    skeleton_tokens: int
    content_type: str  # "python" | "javascript" | "prose" | "mixed"
    compression_ratio: float


def _count_tokens(text: str) -> int:
    """Rough token estimate (≈ word-level)."""
    return len(text.split())


# ── Code skeleton via tree-sitter ─────────────────────────────────────────
_CODE_NODE_TYPES = {
    "python": {
        "keep": {
            "import_statement", "import_from_statement",
            "function_definition", "async_function_definition",
            "class_definition", "decorated_definition",
            "type_alias_statement",
        },
        "body_strip": {"function_definition", "async_function_definition"},
    },
    "javascript": {
        "keep": {
            "import_declaration", "export_statement",
            "function_declaration", "class_declaration",
            "lexical_declaration",
        },
        "body_strip": {"function_declaration"},
    },
}


def _strip_function_body(node, source: bytes, lang: str) -> str:
    """Return function signature + docstring, omitting implementation."""
    lines = source[node.start_byte: node.end_byte].decode("utf-8").splitlines()

    # Keep everything up to (but not including) the first non-header line
    header_lines: list[str] = []
    in_docstring = False
    docstring_done = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if i == 0:
            header_lines.append(line)
            continue
        # Collect docstring
        if not docstring_done:
            if stripped.startswith(('"""', "'''", '`')):
                in_docstring = not in_docstring
                header_lines.append(line)
                if not in_docstring:
                    docstring_done = True
                continue
            if in_docstring:
                header_lines.append(line)
                continue
            if stripped.startswith(("#", "@")):
                header_lines.append(line)
                continue
            # First real body line reached
            docstring_done = True
            header_lines.append("    ...")  # placeholder
            break
        break

    if not docstring_done:
        header_lines.append("    ...")

    return "\n".join(header_lines)


def _skeleton_code(source: str, lang: str) -> str:
    if not _TS_AVAILABLE:
        return _skeleton_code_regex(source, lang)

    language = PY_LANGUAGE if lang == "python" else JS_LANGUAGE
    parser = Parser(language)
    tree = parser.parse(source.encode("utf-8"))
    root = tree.root_node
    src_bytes = source.encode("utf-8")
    cfg = _CODE_NODE_TYPES.get(lang, _CODE_NODE_TYPES["python"])

    parts: list[str] = []
    for child in root.children:
        ntype = child.type
        if ntype not in cfg["keep"]:
            continue
        if ntype in cfg["body_strip"]:
            parts.append(_strip_function_body(child, src_bytes, lang))
        else:
            parts.append(src_bytes[child.start_byte: child.end_byte].decode("utf-8"))

    return "\n\n".join(parts)


def _skeleton_code_regex(source: str, lang: str) -> str:
    """Fallback skeleton for when tree-sitter is unavailable."""
    lines = source.splitlines()
    kept: list[str] = []
    in_body = False
    indent_level = 0

    for line in lines:
        stripped = line.strip()
        # Always keep blank lines, imports, class/def signatures
        if not stripped:
            if not in_body:
                kept.append(line)
            continue
        if stripped.startswith(("import ", "from ", "#!", "//", "/*", "*")):
            kept.append(line)
            in_body = False
            continue
        if re.match(r"^(class|def|async def|function|const|let|var|export)\b", stripped):
            kept.append(line)
            in_body = True
            indent_level = len(line) - len(line.lstrip())
            continue
        if in_body:
            current_indent = len(line) - len(line.lstrip())
            # Keep docstrings at +4 indent
            if current_indent == indent_level + 4 and stripped.startswith(('"""', "'''")):
                kept.append(line)
                continue
            # First real body line → placeholder
            kept.append(" " * (indent_level + 4) + "...")
            in_body = False

    return "\n".join(kept)


# ── Prose skeleton ────────────────────────────────────────────────────────
def _skeleton_prose(text: str) -> str:
    """
    Extract key sentences from prose.
    Strategy: keep first sentence of every paragraph + sentences containing
    high-signal markers (numbers, proper nouns, definitions, etc.).
    """
    paragraphs = re.split(r"\n{2,}", text.strip())
    kept_paragraphs: list[str] = []

    _high_signal = re.compile(
        r"(\d+[\.,]\d+|%|\$|key|important|critical|note|summary|"
        r"conclusion|result|finding|therefore|however|because|due to)",
        re.IGNORECASE,
    )

    for para in paragraphs:
        # Headings, code blocks, tables → keep as-is
        if re.match(r"^(#+|\||```|---)", para.strip()):
            kept_paragraphs.append(para)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        if len(sentences) <= 2:
            kept_paragraphs.append(para)
            continue

        selected = [sentences[0]]  # Always keep first sentence
        for sent in sentences[1:]:
            if _high_signal.search(sent):
                selected.append(sent)

        kept_paragraphs.append(" ".join(selected))

    return "\n\n".join(kept_paragraphs)


# ── Content-type detection ────────────────────────────────────────────────
_LANG_PATTERNS = {
    "python": re.compile(r"(def |class |import |from .+ import|if __name__)"),
    "javascript": re.compile(r"(function |const |let |var |=>|require\(|import )"),
}


def _detect_type(text: str) -> str:
    # If the text has fenced code blocks AND prose paragraphs → mixed
    has_fence = bool(re.search(r"```[\w]*\n", text))
    has_prose_paragraphs = len(re.findall(r"\n\n[A-Z][a-z]", text)) > 1
    if has_fence and has_prose_paragraphs:
        return "mixed"

    # Check for headings (markdown) alongside code
    has_headings = bool(re.search(r"^#{1,3} ", text, re.MULTILINE))
    if has_headings and has_fence:
        return "mixed"

    for lang, pat in _LANG_PATTERNS.items():
        if len(pat.findall(text)) > 3:
            # Only classify as code if prose is sparse
            prose_words = len([w for w in text.split() if w.isalpha()])
            if prose_words < len(text.split()) * 0.5:
                return lang

    code_chars = text.count("{") + text.count("}") + text.count("(")
    if code_chars > 20:
        return "mixed"
    return "prose"


# ── Public API ────────────────────────────────────────────────────────────
def skeletonize(text: str, force_type: Optional[str] = None) -> SkeletonResult:
    """
    Compress text to its structural skeleton.

    Args:
        text:       Input text (code, prose, or mixed).
        force_type: Override detection. One of "python", "javascript", "prose".

    Returns:
        SkeletonResult with the skeleton and stats.
    """
    content_type = force_type or _detect_type(text)
    original_tokens = _count_tokens(text)

    if content_type in ("python", "javascript"):
        skeleton = _skeleton_code(text, content_type)
    elif content_type == "mixed":
        # Split on code fences; skeleton each part separately
        parts = re.split(r"(```[\w]*\n[\s\S]*?```)", text)
        processed: list[str] = []
        for part in parts:
            if part.startswith("```"):
                lang = re.match(r"```(\w*)", part)
                detected = lang.group(1) if lang and lang.group(1) else "python"
                inner = re.sub(r"```\w*\n", "", part).rstrip("`").strip()
                processed.append(f"```{detected}\n{_skeleton_code(inner, detected)}\n```")
            else:
                processed.append(_skeleton_prose(part) if part.strip() else part)
        skeleton = "".join(processed)
    else:
        skeleton = _skeleton_prose(text)

    skeleton_tokens = _count_tokens(skeleton)
    ratio = original_tokens / max(skeleton_tokens, 1)

    return SkeletonResult(
        skeleton=skeleton,
        original_tokens=original_tokens,
        skeleton_tokens=skeleton_tokens,
        content_type=content_type,
        compression_ratio=round(ratio, 2),
    )
