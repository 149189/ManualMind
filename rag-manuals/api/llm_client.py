import requests, os

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8080")

def query_llm(prompt: str, max_tokens: int = 512):
    resp = requests.post(
        f"{LLM_API_URL}/generate",
        json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}},
        timeout=60
    )
    return resp.json().get("generated_text", "").strip()
