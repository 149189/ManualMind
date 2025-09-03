# api/prompt_templates.py
"""
Prompt templates and helpers for ManualMind.

This module formats retrieved snippets (with per-snippet scores) and
builds an instruction prompt that asks the LLM to:

  - weigh snippets according to their retrieval score,
  - compose a concise, human-style summarized answer,
  - explicitly call out uncertainty / contradictions,
  - emit a JSON object with `answer`, `citations`, and `llm_confidence`.

The prompt limits snippet size to avoid extremely long contexts.
"""

from typing import List, Tuple, Dict, Any
import html
import textwrap

# -------------------------
# System / instruction text
# -------------------------
QUERY_SYSTEM = (
    "You are an assistant whose job is to answer user questions using ONLY the provided "
    "retrieved snippets from product manuals. Do not use any external knowledge beyond the snippets. "
    "Be concise, factual, and explicit about uncertainty. When snippets conflict or are low-confidence, "
    "say so clearly and avoid inventing facts."
)

# -------------------------
# Helpers
# -------------------------
def truncate_text(text: str, max_chars: int = 1000) -> str:
    """
    Truncate a text to a safe length without breaking words aggressively.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    # Keep whole words when possible
    truncated = textwrap.shorten(text, width=max_chars, placeholder=" ...")
    return truncated

# -------------------------
# Formatting functions
# -------------------------
def format_retrieved_snippets(results: List[Tuple[float, Dict[str, Any]]], max_snippet_chars: int = 1200) -> str:
    """
    Convert FAISS results (score, metadata) into a numbered snippet block.

    Args:
        results: list of tuples (score: float, metadata: dict) where metadata includes
                 keys like 'text', 'source', 'page', 'id' (optional).
        max_snippet_chars: maximum characters to include per snippet (truncation).

    Returns:
        A single string containing numbered snippets labelled S1, S2, ...
        Each snippet includes: ID, score (0..1), source, page, and truncated text.
    """
    lines = []
    for i, (score, meta) in enumerate(results, start=1):
        sid = f"S{i}"
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        chunk_id = meta.get("id", meta.get("chunk_id", ""))
        raw_text = meta.get("text", "")
        safe_text = truncate_text(raw_text, max_snippet_chars)
        # escape problematic characters (keeps JSON/parse friendly)
        safe_text = safe_text.replace("\n", " ").strip()
        lines.append(f"{sid} | score: {score:.4f} | source: {source} | page: {page} | chunk_id: {chunk_id}\n{safe_text}")
    return "\n\n".join(lines)


def build_answer_prompt(question: str, results: List[Tuple[float, Dict[str, Any]]],
                        max_context_chars: int = 6000) -> str:
    """
    Build the full prompt to send to the LLM. It includes:
      - a short system instruction (QUERY_SYSTEM),
      - the question,
      - the formatted snippets (with scores),
      - explicit instruction to weigh snippets by score,
      - required JSON output format.

    Args:
        question: user's question string.
        results: list of (score, metadata) tuples returned from FaissStore.search.
        max_context_chars: approximate max number of characters for the context block (snippets).
                           The function will keep adding snippets until the limit is reached.

    Returns:
        A string prompt ready to be sent to the LLM.
    """
    # Build snippet list but limit total size to max_context_chars
    snippet_texts = []
    total_chars = 0
    for i, (score, meta) in enumerate(results, start=1):
        # Truncate each snippet to a reasonable size
        snippet = format_retrieved_snippets([(score, meta)], max_snippet_chars=1200)
        snippet_len = len(snippet)
        if total_chars + snippet_len > max_context_chars and snippet_texts:
            # stop adding more snippets if context budget exceeded (keep at least one)
            break
        snippet_texts.append(snippet)
        total_chars += snippet_len

    snippets_block = "\n\n".join(snippet_texts) if snippet_texts else "(no snippets available)"

    # Instruction block: emphasize weighing by score and not inventing facts.
    instructions = (
        "Instructions for the assistant:\n"
        "1) Use ONLY the information found in the provided snippets. Do NOT add external facts.\n"
        "2) Each snippet has a retrieval score (0.0 - 1.0). Treat higher scores as stronger evidence.\n"
        "   Weight your reasoning by these scores: prioritize high-score snippets when they contain "
        "   overlapping facts; if only low-score snippets exist, indicate low confidence.\n"
        "3) If snippets contradict each other, state the contradiction and present both possibilities, "
        "   indicating which snippet(s) support each view.\n"
        "4) Produce a concise, user-facing SUMMARY answer (1-5 short paragraphs). Be direct and practical.\n"
        "5) After the summary, output a JSON object ONLY (no additional text) with the keys:\n"
        "   {\n"
        "     \"answer\": \"<your concise answer string>\",\n"
        "     \"citations\": [\"S1\", \"S2\", ...],       # list only snippet IDs you relied on\n"
        "     \"llm_confidence\": <number 0-100>        # your estimated confidence percentage\n"
        "   }\n"
        "   The `answer` string should be human readable; `citations` point to snippet IDs above.\n"
        "6) If you are uncertain or cannot answer from the snippets, set `answer` to a short honesty sentence "
        "   (e.g. \"I don't know based on the provided manuals.\") and set `llm_confidence` low (e.g., 0-30).\n"
    )

    # Compose final prompt
    prompt_parts = [
        QUERY_SYSTEM,
        "",
        "Question:",
        question,
        "",
        "Retrieved snippets (each labeled S1, S2, ... with a score between 0.0 and 1.0):",
        snippets_block,
        "",
        instructions,
        "",
        "Now produce the summarized answer and the JSON object as described above."
    ]

    prompt = "\n".join(prompt_parts)
    return prompt
