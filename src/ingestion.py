"""
src/ingestion.py — Phase 1 (and ongoing) PDF ingestion into Qdrant.

Key design change from the original version: ingestion is now per-document
and idempotent (delete-then-upsert by doc_id), not a single collection-wide
force_recreate. This means:

  - Re-running ingestion is safe and does NOT wipe unrelated documents.
  - A single new/updated PDF can be ingested with --file, touching only
    that document's chunks.
  - A document can be removed with --delete-doc <doc_id> without touching
    anything else.

This directly supports the "can I add/update/delete documents later?"
requirement — previously the answer was no (force_recreate wiped everything).

Every chunk's payload now carries doc_id (content hash of the PDF bytes —
the stable identity used for delete/update targeting), scope (matches
src/retrieval.py's DEFAULT_SCOPE = "base"; this is also what lets Phase 10's
upload-mode documents later coexist in the same collection under a different
scope without a schema change), and chunk_index (position within the
document, useful for debugging/ordering). doc_name and page remain exactly
as before — src/retrieval.py's filtering/display logic is unchanged.
"""

import argparse
import hashlib
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient, models

from src.config import config

DATA_DIR = "./data/raw_pdfs"
RESULTS_DIR = "./results"

# Must match src.retrieval.DEFAULT_SCOPE. Duplicated as a plain constant
# (not imported) to keep ingestion.py importable standalone inside a
# container without pulling in retrieval.py's heavier dependencies
# (reranker model, BM25). If you change one, change the other.
DEFAULT_SCOPE = "base"

# Deterministic namespace for deriving valid Qdrant point UUIDs from
# (doc_id, chunk_index) pairs. Any fixed UUID works; this one is arbitrary
# but must never change, or re-ingesting the same document would produce
# different point IDs and silently duplicate rather than update.
POINT_ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def compute_doc_id(pdf_path: Path) -> str:
    """Content-based document identity (hash of the raw PDF bytes), used as
    the stable key for delete/upsert targeting. Two files with different
    names but identical content get the same doc_id (treated as the same
    document); re-ingesting the same file always produces the same doc_id,
    making ingestion idempotent."""
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def compute_point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic, valid Qdrant point ID (UUID5) for a given chunk.
    Re-ingesting the same document produces the same point IDs, so an
    upsert naturally overwrites in place rather than duplicating."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"{doc_id}:{chunk_index}"))


def build_client() -> QdrantClient:
    """Same local-vs-server branching logic as src.retrieval.Retriever._build_client.
    Kept duplicated rather than imported for the same standalone-container
    reason as DEFAULT_SCOPE above."""
    if config.qdrant_use_local:
        return QdrantClient(path=config.qdrant_local_path)
    return QdrantClient(url=f"http://{config.qdrant_host}:{config.qdrant_port}")


def ensure_collection(client: QdrantClient, embeddings: FastEmbedEmbeddings) -> None:
    """Creates the collection if it doesn't exist yet. Vector size is
    determined by actually embedding a short string, not hardcoded — so
    this doesn't silently break if the embedding model is ever changed."""
    existing = [c.name for c in client.get_collections().collections]
    if config.qdrant_collection_name in existing:
        return

    probe_vector = embeddings.embed_query("dimension probe")
    print(f"[INFO] Creating collection '{config.qdrant_collection_name}' "
          f"(vector size={len(probe_vector)}, distance=Cosine)...")
    client.create_collection(
        collection_name=config.qdrant_collection_name,
        vectors_config=models.VectorParams(
            size=len(probe_vector),
            distance=models.Distance.COSINE,
        ),
    )


def load_single_pdf(path: Path) -> Tuple[List, Optional[str]]:
    """Loads one PDF's pages via pdfplumber. Returns (pages, error_or_None)."""
    try:
        loader = PDFPlumberLoader(str(path))
        docs = loader.load()
        for doc in docs:
            # PDFPlumberLoader indexes pages from 0; normalize to 1-indexed
            # to match eval set page numbering and avoid a "Page: 0" citation.
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        return docs, None
    except Exception as e:
        return [], str(e)


def split_documents(documents: List) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return text_splitter.split_documents(documents)


def delete_document(client: QdrantClient, doc_id: str, scope: str = DEFAULT_SCOPE) -> int:
    """Deletes all chunks belonging to doc_id within scope. Returns nothing
    countable from Qdrant's delete response directly, so this just issues
    the delete; call count_document() before/after if you need the number
    removed. Safe to call on a doc_id with zero existing chunks (no-op)."""
    client.delete(
        collection_name=config.qdrant_collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id)),
                    models.FieldCondition(key="metadata.scope", match=models.MatchValue(value=scope)),
                ]
            )
        ),
    )
    return 0


def ingest_single_pdf(
    client: QdrantClient,
    embeddings: FastEmbedEmbeddings,
    path: Path,
    scope: str = DEFAULT_SCOPE,
) -> Dict:
    """
    Ingests exactly one PDF: delete any existing chunks for this doc_id
    (makes re-ingestion of an updated file safe/idempotent), then embed
    and upsert the new chunks. Does NOT touch any other document's chunks.

    Returns a dict with per-file stats for the report.
    """
    print(f"Loading: {path.name}...")
    pages, error = load_single_pdf(path)
    if error:
        print(f"  -> Failed to load {path.name}: {error}")
        return {"file": path.name, "ok": False, "error": error, "pages": 0, "chunks": 0}

    for p in pages:
        p.metadata["doc_name"] = path.name

    chunks = split_documents(pages)
    if not chunks:
        print(f"  -> {path.name} produced 0 chunks (empty or unparseable), skipping.")
        return {"file": path.name, "ok": False, "error": "0 chunks produced", "pages": len(pages), "chunks": 0}

    doc_id = compute_doc_id(path)

    # Idempotent re-ingestion: clear this document's previous chunks first.
    # Safe on first ingest too (deletes nothing, since no prior chunks exist).
    delete_document(client, doc_id, scope)

    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(models.PointStruct(
            id=compute_point_id(doc_id, idx),
            vector=vector,
            payload={
                "page_content": chunk.page_content,
                "metadata": {
                    "doc_name": path.name,
                    "doc_id": doc_id,
                    "page": chunk.metadata.get("page", 0),
                    "chunk_index": idx,
                    "scope": scope,
                },
            },
        ))

    client.upsert(collection_name=config.qdrant_collection_name, points=points)
    print(f"  -> {path.name}: {len(pages)} pages, {len(chunks)} chunks ingested (doc_id={doc_id}).")

    return {"file": path.name, "ok": True, "error": None, "pages": len(pages), "chunks": len(chunks), "doc_id": doc_id}


def write_ingestion_report(results: List[Dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "ingestion_report.md")

    total = len(results)
    successful = sum(1 for r in results if r["ok"])
    total_pages = sum(r["pages"] for r in results)
    total_chunks = sum(r["chunks"] for r in results)
    success_rate = (successful / total * 100) if total > 0 else 0.0
    passed_threshold = success_rate >= 90.0

    lines = [
        "# Ingestion Run Report\n",
        f"**Total PDFs Attempted:** {total}",
        f"**Successfully Loaded:** {successful}",
        f"**Total Pages Parsed:** {total_pages}",
        f"**Total Vector Chunks:** {total_chunks}",
        f"**Success Rate:** {success_rate:.1f}%\n",
    ]

    if passed_threshold:
        lines.append("## Status: PASSED")
        lines.append("Ingestion met the >= 90% success threshold.\n")
    else:
        lines.append("## Status: FAILED")
        lines.append("Ingestion failed to meet the 90% success threshold.\n")

    failures = [r for r in results if not r["ok"]]
    if failures:
        lines.append("### Failed Files:")
        for r in failures:
            lines.append(f"- **{r['file']}**: `{r['error']}`")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport written to: {report_path}")
    if not passed_threshold:
        print(f"[WARNING] Ingestion failed threshold check ({success_rate:.1f}% < 90%). See report for details.")


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant (idempotent, per-document).")
    parser.add_argument("--file", type=str, default=None,
                         help="Ingest a single PDF by path instead of the whole data/raw_pdfs/ directory.")
    parser.add_argument("--delete-doc", type=str, default=None,
                         help="Delete all chunks for the given doc_id and exit (no ingestion).")
    parser.add_argument("--scope", type=str, default=DEFAULT_SCOPE,
                         help=f"Scope to ingest into/delete from (default: {DEFAULT_SCOPE}).")
    args = parser.parse_args()

    print("Initializing FastEmbed embeddings (lightweight ONNX runtime)...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    client = build_client()
    ensure_collection(client, embeddings)

    if args.delete_doc:
        print(f"Deleting all chunks for doc_id={args.delete_doc} (scope={args.scope})...")
        delete_document(client, args.delete_doc, args.scope)
        print("Done.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.file:
        pdf_paths = [Path(args.file)]
        if not pdf_paths[0].exists():
            print(f"[ERROR] File not found: {args.file}")
            return
    else:
        pdf_paths = sorted(Path(DATA_DIR).glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in {DATA_DIR}.")
        write_ingestion_report([])
        return

    results = [ingest_single_pdf(client, embeddings, p, args.scope) for p in pdf_paths]

    print(f"\nTotal: {sum(r['pages'] for r in results)} pages, "
          f"{sum(r['chunks'] for r in results)} chunks across {len(results)} file(s).")

    write_ingestion_report(results)


if __name__ == "__main__":
    main()