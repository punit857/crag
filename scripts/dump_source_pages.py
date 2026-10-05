"""
Dumps the real extracted text of every PDF, page by page, to a single text file
so eval questions can be written against actual content.

Uses pdfplumber (same extractor as ingestion) so the text matches what the
retriever sees, including page numbering (1-indexed).
"""
import sys
from pathlib import Path

import pdfplumber

# Adjust if your PDFs live elsewhere. Falls back to a recursive search.
CANDIDATE_DIRS = [Path("data") / "raw_pdfs", Path("data") / "pdfs", Path("data"), Path(".")]
OUT_PATH = Path("results") / "source_pages_dump.txt"


def find_pdfs():
    for d in CANDIDATE_DIRS:
        if d.exists():
            pdfs = sorted(p for p in d.rglob("*.pdf") if ".venv" not in p.parts)
            if pdfs:
                print(f"[INFO] Found {len(pdfs)} PDFs under {d}")
                return pdfs
    return []


def main():
    pdfs = find_pdfs()
    if not pdfs:
        sys.exit("[ERROR] No PDFs found. Tell me where they live.")

    Path("results").mkdir(exist_ok=True)
    total_pages = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for pdf_path in pdfs:
            with pdfplumber.open(pdf_path) as pdf:
                out.write(f"\n\n{'#' * 100}\n# DOC: {pdf_path.name}  ({len(pdf.pages)} pages)\n{'#' * 100}\n")
                for i, page in enumerate(pdf.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    out.write(f"\n===== {pdf_path.name} | PAGE {i} =====\n{text}\n")
                    total_pages += 1
            print(f"[OK] {pdf_path.name}")
    print(f"\n[INFO] {total_pages} pages written to {OUT_PATH}")


if __name__ == "__main__":
    main()