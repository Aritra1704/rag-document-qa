"""Utilities for local Ollama-based RAG workflows."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Sequence

import chromadb


def _normalize_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def _get_json(base_url: str, path: str, timeout: int = 10) -> dict[str, Any]:
    url = f"{_normalize_base_url(base_url)}{path}"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc

    if not payload.strip():
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}") from exc


def _post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: int = 60,
) -> dict[str, Any]:
    url = f"{_normalize_base_url(base_url)}{path}"
    request_body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc

    if not response_body.strip():
        return {}
    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}") from exc


def list_ollama_models(base_url: str) -> list[str]:
    payload = _get_json(base_url, "/api/tags", timeout=10)
    models = payload.get("models", [])
    names: list[str] = []
    for model in models:
        if isinstance(model, dict) and model.get("name"):
            names.append(str(model["name"]))
    return names


def _model_available(model_names: Sequence[str], target_model: str) -> bool:
    target = target_model.strip().lower()
    if not target:
        return False

    normalized = [name.strip().lower() for name in model_names]
    if target in normalized:
        return True

    target_base = target.split(":")[0]
    return any(name.split(":")[0] == target_base for name in normalized)


def get_ollama_diagnostics(
    base_url: str,
    embedding_model: str,
    chat_model: str,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "endpoint_reachable": False,
        "endpoint_error": None,
        "model_names": [],
        "embedding_model_available": False,
        "chat_model_available": False,
    }

    try:
        model_names = list_ollama_models(base_url)
    except Exception as exc:  # noqa: BLE001
        diagnostics["endpoint_error"] = str(exc)
        return diagnostics

    diagnostics["endpoint_reachable"] = True
    diagnostics["model_names"] = model_names
    diagnostics["embedding_model_available"] = _model_available(model_names, embedding_model)
    diagnostics["chat_model_available"] = _model_available(model_names, chat_model)
    return diagnostics


class OllamaEmbeddingFunction:
    """Chroma-compatible embedding function backed by local Ollama."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = _normalize_base_url(base_url)
        self.model_name = model_name.strip()

    def _embed_with_embeddings_endpoint(self, text: str) -> list[float]:
        response = _post_json(
            self.base_url,
            "/api/embeddings",
            {"model": self.model_name, "prompt": text},
            timeout=120,
        )
        embedding = response.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Missing embedding vector in /api/embeddings response")
        return [float(value) for value in embedding]

    def _embed_with_embed_endpoint(self, text: str) -> list[float]:
        response = _post_json(
            self.base_url,
            "/api/embed",
            {"model": self.model_name, "input": text},
            timeout=120,
        )
        embeddings = response.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return [float(value) for value in embeddings[0]]
        embedding = response.get("embedding")
        if isinstance(embedding, list):
            return [float(value) for value in embedding]
        raise RuntimeError("Missing embedding vector in /api/embed response")

    def embed_text(self, text: str) -> list[float]:
        primary_error: str | None = None
        try:
            return self._embed_with_embeddings_endpoint(text)
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)

        try:
            return self._embed_with_embed_endpoint(text)
        except Exception as exc:  # noqa: BLE001
            fallback_error = str(exc)
            raise RuntimeError(
                f"Ollama embedding failed via /api/embeddings ({primary_error}) "
                f"and /api/embed ({fallback_error})"
            ) from exc

    def __call__(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002
        return [self.embed_text(str(item)) for item in input]


def create_ollama_collection(
    base_url: str,
    embedding_model: str,
    collection_name: str = "ollama_documents",
):
    client = chromadb.Client()
    try:
        client.delete_collection(name=collection_name)
    except Exception:  # noqa: BLE001
        pass

    embedding_function = OllamaEmbeddingFunction(base_url=base_url, model_name=embedding_model)
    return client.create_collection(name=collection_name, embedding_function=embedding_function)


def generate_ollama_answer(
    base_url: str,
    chat_model: str,
    query: str,
    context_chunks: Sequence[str],
    context_metadata: Sequence[dict[str, Any]],
) -> str:
    context_lines: list[str] = []
    for chunk, metadata in zip(context_chunks, context_metadata):
        source = str(metadata.get("source", "unknown"))
        page = metadata.get("page")
        if page in (None, "N/A", "", 0):
            source_label = source
        else:
            source_label = f"{source} (page {page})"
        context_lines.append(f"[Source: {source_label}]\n{chunk}")

    context = "\n\n".join(context_lines)
    prompt = (
        "You are a local document assistant.\n"
        "Use ONLY the provided context to answer the question.\n"
        "If the context is insufficient, say that clearly.\n"
        "Reference sources from the context in your answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = _post_json(
        base_url,
        "/api/chat",
        {
            "model": chat_model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=180,
    )
    message = response.get("message", {})
    content = message.get("content")
    if not content:
        raise RuntimeError("Ollama chat response did not contain answer content")
    return str(content)
