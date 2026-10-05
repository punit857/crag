import hashlib
from typing import Dict, List, Optional

from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from rank_bm25 import BM25Okapi
from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.config import config

# Default scope for all current (non-upload-mode) data. Every chunk written
# by ingestion.py should carry payload["metadata"]["scope"] = DEFAULT_SCOPE.
# This exists now so the Phase 10 "upload your own document" feature (session-
# scoped data living alongside the base knowledge base in one collection) can
# be added later without a schema migration. Nothing outside Phase 10 should
# ever need a different scope value yet.
DEFAULT_SCOPE = "base"

# How many points to fetch per page when loading the full chunk set for BM25.
# The original code used a single scroll(limit=10000) call with no pagination
# fallback — fine at 170 chunks, silently wrong once the corpus grows past
# whatever Qdrant's effective page cap turns out to be. Paginate properly now.
SCROLL_PAGE_SIZE = 1000


def get_chunk_id(doc_name: str, page: int, text: str) -> str:
    """
    Deterministic content-based ID for a chunk. Used consistently across
    BOTH dense_search and bm25_search so RRF fusion can correctly recognize
    the same chunk when it's returned by both retrieval methods.

    IMPORTANT: Do NOT trust any '_id' injected into search-result metadata
    by langchain-qdrant — that id scheme does not match what a raw
    QdrantClient.scroll() call sees, and mixing the two breaks RRF dedup.
    """
    content = f"{doc_name}_{page}_{text[:50]}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


class Retriever:
    def __init__(self, scope: str = DEFAULT_SCOPE):
        """
        scope: filters which chunks this Retriever instance sees, via a
        Qdrant payload filter on metadata.scope. Defaults to DEFAULT_SCOPE
        ("base"), matching all current ingestion. Not yet exposed for
        per-request override — that's Phase 10 (upload mode) work.
        """
        self.scope = scope
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.client = self._build_client()
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=config.qdrant_collection_name,
            embedding=self.embeddings,
        )
        self._scope_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.scope",
                    match=models.MatchValue(value=self.scope),
                )
            ]
        )

        self.reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
        self._build_bm25_index()  # sets self.all_chunks and self.bm25

    def _build_client(self) -> QdrantClient:
        """
        FIX: previously always connected via local-file mode, ignoring
        config.qdrant_use_local entirely. Now honors the toggle, so the same
        code runs against local-file Qdrant (dev) or the Docker server
        (config.qdrant_use_local = False) without any code change — only
        config changes.
        """
        if config.qdrant_use_local:
            return QdrantClient(path=config.qdrant_local_path)
        return QdrantClient(url=f"http://{config.qdrant_host}:{config.qdrant_port}")

    def _build_bm25_index(self) -> None:
        """Loads all chunks in this Retriever's scope and (re)builds the BM25 index."""
        self.all_chunks = self._load_all_chunks()
        tokenized_corpus = [chunk["text"].lower().split() for chunk in self.all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def refresh(self) -> None:
        """
        Rebuilds the in-memory chunk list and BM25 index from Qdrant.

        BM25 is built once at startup and never updates itself when the
        collection changes underneath it — a long-lived process (e.g. the
        FastAPI server) would otherwise keep answering BM25 queries against
        stale data after any re-ingestion. Call this after any ingestion/
        upsert/delete operation that should be reflected in search results.
        Dense search doesn't need this (it queries Qdrant live on every call).
        """
        self._build_bm25_index()

    def _load_all_chunks(self) -> List[Dict]:
        """
        Fetch all chunks (within this Retriever's scope) from Qdrant to build
        the BM25 sparse index.

        FIX: paginated via next_page_offset instead of a single
        limit=10000 call — the original approach would have silently
        truncated the BM25 index once the collection grew past whatever
        Qdrant's effective single-page cap is, with no error, just missing
        chunks from sparse search. Correct at any corpus size now.
        """
        chunks: List[Dict] = []
        offset: Optional[models.ExtendedPointId] = None

        while True:
            records, next_offset = self.client.scroll(
                collection_name=config.qdrant_collection_name,
                scroll_filter=self._scope_filter,
                limit=SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                payload = r.payload or {}
                metadata = payload.get("metadata", {})
                text = payload.get("page_content", "")
                doc_name = metadata.get("doc_name", "")
                page = metadata.get("page", 0)
                chunk_id = get_chunk_id(doc_name, page, text)
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "doc_name": doc_name,
                    "page": page,
                })

            if next_offset is None:
                break
            offset = next_offset

        return chunks

    def dense_search(self, query: str, top_k: int = 10) -> List[Dict]:
        # NOTE: the `filter` kwarg below is langchain-qdrant's documented way
        # to pass a Qdrant payload filter through to the underlying search.
        # This is a new code path not yet exercised — if this raises a
        # TypeError about an unexpected 'filter' keyword, that means this
        # langchain-qdrant version's signature differs from what's documented;
        # report that immediately rather than assuming it silently worked.
        results = self.vector_store.similarity_search_with_score(
            query, k=top_k, filter=self._scope_filter
        )
        formatted = []
        for doc, score in results:
            text = doc.page_content
            doc_name = doc.metadata.get("doc_name", "")
            page = doc.metadata.get("page", 0)
            chunk_id = get_chunk_id(doc_name, page, text)
            formatted.append({
                "id": chunk_id,
                "text": text,
                "doc_name": doc_name,
                "page": page,
                "score": score,
            })
        return formatted

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.bm25 or not self.all_chunks:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_n_idx:
            chunk = self.all_chunks[idx].copy()
            chunk["score"] = scores[idx]
            results.append(chunk)
        return results

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Hybrid Search using Reciprocal Rank Fusion (BM25 + Dense)."""
        candidate_k = 20
        dense_res = self.dense_search(query, top_k=candidate_k)
        bm25_res = self.bm25_search(query, top_k=candidate_k)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict] = {}
        k_constant = getattr(config, "rrf_k_constant", 60)

        for rank, chunk in enumerate(dense_res):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_constant + rank + 1))
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(bm25_res):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_constant + rank + 1))
            chunk_map[cid] = chunk

        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for cid in sorted_cids[:top_k]:
            c = chunk_map[cid].copy()
            c["score"] = rrf_scores[cid]
            results.append(c)
        return results

    def reranked_hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Hybrid Search (RRF) + Cross-Encoder Reranking."""
        candidates = self.hybrid_search(query, top_k=20)
        if not candidates:
            return []

        docs = [c["text"] for c in candidates]
        rerank_scores = list(self.reranker.rerank(query, docs))

        for i, c in enumerate(candidates):
            c["score"] = float(rerank_scores[i])

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]