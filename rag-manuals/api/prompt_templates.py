# rag-manuals/api/prompt_templates.py


QUERY_SYSTEM = """
You are ManualMind, an assistant that MUST answer only from the provided snippets. For each factual claim, add an inline citation like (S1: page 5).
If the answer is not present in the snippets, reply with: "I don't know based on the provided manuals."
Return a JSON object with fields: answer (string), citations (list of snippet ids), llm_confidence (0-100).
"""




def format_retrieved_snippets(retrieved):
	# retrieved: List[(score, meta)]
	parts = []
	for i, (score, meta) in enumerate(retrieved, start=1):
		sid = f"S{i}"
		header = f"[{sid}] {meta.get('source','unknown')} :: page {meta.get('page','?')} :: score={score:.4f}"
		parts.append(header)
		parts.append(meta.get('text',''))
	return "\n\n".join(parts)