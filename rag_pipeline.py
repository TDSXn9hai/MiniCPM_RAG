"""Offline retrieval and answer orchestration, independent of model imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable

from rag_core import (
    EvidenceCard, KnowledgeBase, KnowledgeDocument, RetrievalHit, build_evidence_card,
)

REFUSAL = "当前知识库没有足够依据回答这个问题，请补充相关资料。"


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 3
    min_score: float = 0.08
    max_evidence_chars: int = 900
    min_shared_bigrams: int = 2
    title_only: bool = False

    def __post_init__(self):
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if self.max_evidence_chars < 1:
            raise ValueError("max_evidence_chars must be positive")
        if self.min_shared_bigrams < 1:
            raise ValueError("min_shared_bigrams must be positive")


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    hits: tuple[RetrievalHit, ...]
    evidence: EvidenceCard
    knowledge_path: str
    knowledge_sha256: str
    retrieval_ms: float

    @property
    def usable(self):
        return bool(self.evidence.text.strip() and self.evidence.citations)

    def to_dict(self):
        return {
            "query": self.query,
            "knowledge_path": self.knowledge_path,
            "knowledge_sha256": self.knowledge_sha256,
            "retrieval_ms": self.retrieval_ms,
            "usable": self.usable,
            "hits": [
                {"document_id": h.document.id, "title": h.document.title,
                 "source": h.document.source, "score": h.score}
                for h in self.hits
            ],
            "evidence": self.evidence.text,
            "citations": [asdict(c) for c in self.evidence.citations],
        }


class RAGRetriever:
    """Rebuild the tiny offline index when knowledge file contents change."""

    def __init__(self, knowledge_path, config: RetrievalConfig | None = None):
        self.path = Path(knowledge_path).expanduser().resolve()
        self.config = config or RetrievalConfig()
        self._knowledge_base = None
        self._sha256 = ""

    def _reload(self):
        raw = self.path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest == self._sha256:
            return
        payload = json.loads(raw)
        items = payload.get("documents") if isinstance(payload, dict) else None
        if not isinstance(items, list) or not items:
            raise ValueError("knowledge JSON must contain a nonempty documents list")
        documents, seen = [], set()
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"document {index} must be an object")
            for field in ("id", "title", "content"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    raise ValueError(f"document {index}: {field} must be a nonempty string")
            identity = item["id"].strip()
            if identity in seen:
                raise ValueError(f"duplicate document id: {identity}")
            seen.add(identity)
            if not isinstance(item.get("source", ""), str):
                raise ValueError(f"document {index}: source must be a string")
            documents.append(KnowledgeDocument(
                identity, item["title"].strip(), item["content"].strip(),
                item.get("source", "").strip(),
            ))
        # Commit both only after parsing and indexing succeed.
        self._knowledge_base = KnowledgeBase.from_documents(
            documents, title_only=self.config.title_only
        )
        self._sha256 = digest

    def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        start = time.perf_counter()
        self._reload()
        candidates = self._knowledge_base.search(
            query, top_k=len(self._knowledge_base.documents), min_score=self.config.min_score
        )
        def bigrams(text):
            normalized = re.sub(r"\W+", "", text.casefold())
            return {normalized[i:i + 2] for i in range(len(normalized) - 1)}
        query_grams = bigrams(query)
        # A TF-IDF query's unknown terms are dropped from its vector. A single
        # shared generic word such as “实验” can therefore look falsely strong.
        # Retain the old recall threshold, but require two distinct matching
        # bigrams for longer questions (one for a two-character query).
        required = min(self.config.min_shared_bigrams, len(query_grams))
        hits = tuple(
            h for h in candidates
            if query_grams and len(query_grams & bigrams(
                h.document.title if self.config.title_only else
                h.document.title + h.document.content
            )) >= required
        )[:self.config.top_k]
        card = build_evidence_card(hits, max_chars=self.config.max_evidence_chars)
        return RetrievalResult(
            query.strip(), hits, card, str(self.path), self._sha256,
            (time.perf_counter() - start) * 1000,
        )


class RAGPipeline:
    """One synchronous owner calls the runtime; no background model threads."""

    def __init__(self, retriever: RAGRetriever, runtime):
        self.retriever = retriever
        self.runtime = runtime

    def answer(self, query: str, *, audio=None, on_chunk: Callable | None = None):
        return self.answer_retrieved(
            self.retriever.retrieve(query), audio=audio, on_chunk=on_chunk
        )

    def answer_retrieved(
        self, retrieval: RetrievalResult, *, audio=None, on_chunk: Callable | None = None
    ):
        result = {
            "query": retrieval.query,
            "retrieval": retrieval.to_dict(),
            "citations": [asdict(c) for c in retrieval.evidence.citations],
            "units": [],
        }
        if not retrieval.usable:
            return {**result, "status": "no_evidence", "text": REFUSAL,
                    "ended_naturally": True, "model_called": False}
        if self.runtime is None:
            raise RuntimeError("an answer runtime is required")
        text = []
        end_of_turn = False
        stream = self.runtime.stream_answer(
            query=retrieval.query, evidence=retrieval.evidence.text, audio=audio
        )
        try:
            for chunk in stream:
                if on_chunk is not None:
                    on_chunk(chunk)
                text.append(chunk.get("text") or "")
                end_of_turn = bool(chunk["end_of_turn"])
                # Audio goes to the consumer, never duplicated in JSON.
                result["units"].append({
                    k: v for k, v in chunk.items() if k != "audio"
                })
        finally:
            # Consumer failures must release the runtime's active session too.
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        joined = "".join(text)
        status = "answered" if end_of_turn and joined.strip() else "incomplete"
        return {**result, "status": status, "text": joined,
                "ended_naturally": end_of_turn, "model_called": True}
