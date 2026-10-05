"""
fetch_corpus.py — Downloads a small, free, public-domain corpus of
short DOE industrial-equipment-efficiency technical documents
(tip sheets on pumps, compressed air, steam, and motor systems)
for the CRAG knowledge base.

Uses a hardcoded list of individually-verified direct PDF URLs —
NOT scraping — because energy.gov's dynamic "node" listing pages
proved unreliable (inconsistent 404s) even though the static PDF
files themselves download fine.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from pypdf import PdfReader

OUTPUT_DIR = Path("data/raw_pdfs")
REPORT_PATH = Path("results/corpus_fetch_report.md")

# Individually verified direct PDF URLs — DOE AMO / Industrial Technologies
# Program tip sheets and short technical briefs. Each confirmed to exist
# via direct fetch before being added here.
PDF_URLS = [
    "https://www.energy.gov/sites/prod/files/2014/05/f16/compressed_air2.pdf",
    "https://www.energy.gov/sites/prod/files/2014/04/f15/39157.pdf",
    "https://www.energy.gov/sites/prod/files/2014/05/f16/match_pumps_to_system.pdf",
    "https://www.energy.gov/sites/prod/files/2014/05/f16/test_pumping_system__pumping_systemts4.pdf",
    "https://www1.eere.energy.gov/manufacturing/tech_assistance/pdfs/compressed_air11.pdf",
    "https://www1.eere.energy.gov/manufacturing/tech_assistance/pdfs/compressed_air10.pdf",
    "https://www1.eere.energy.gov/manufacturing/tech_assistance/pdfs/39156.pdf",
    "https://www1.eere.energy.gov/manufacturing/tech_assistance/pdfs/steam21_rotating_equip.pdf",
    "https://www1.eere.energy.gov/manufacturing/tech_assistance/pdfs/heatpump.pdf",
    "https://www1.eere.energy.gov/manufacturing/tech_assistance/pdfs/metal_cs_technicast.pdf",
    "https://www1.eere.energy.gov/manufacturing/tech_assistance/pdfs/fujifilm_case_study.pdf",
    "https://www.energy.gov/sites/prod/files/2014/05/f16/us_industrial_motor_driven.pdf",
    "https://www.energy.gov/sites/prod/files/2014/05/f16/market_assessment_glimpse.pdf",
    "https://www1.eere.energy.gov/manufacturing/tech_assistance/pdfs/mi_cs_ashgrove.pdf",
]

MAX_PAGES_PER_DOC = 15   # heatpump.pdf is a longer technical brief, not a 2-page tip sheet
MAX_TOTAL_PAGES = 150
REQUEST_TIMEOUT = 30
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def slugify(url: str) -> str:
    return url.split("/")[-1]


def download_pdf(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"[SKIP] already downloaded: {dest.name}")
        return True
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                print(f"[FAIL] not a valid PDF: {url}")
                return False
            dest.write_bytes(resp.content)
            return True
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"[RETRY {attempt}/3] {url} — {e} (waiting {wait}s)")
            time.sleep(wait)
    print(f"[FAIL] giving up on: {url}")
    return False


def get_page_count(path: Path) -> Optional[int]:
    try:
        return len(PdfReader(str(path)).pages)
    except Exception as e:
        print(f"[WARN] could not read {path.name}: {e}")
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    accepted: List[Tuple[str, int, str]] = []
    skipped: List[Tuple[str, str]] = []
    total_pages = 0

    for url in PDF_URLS:
        dest = OUTPUT_DIR / slugify(url)

        if not download_pdf(url, dest):
            skipped.append((url, "download failed"))
            continue

        pages = get_page_count(dest)
        if pages is None:
            skipped.append((url, "unreadable PDF"))
            dest.unlink(missing_ok=True)
            continue

        if pages > MAX_PAGES_PER_DOC:
            skipped.append((url, f"too long ({pages} pages > {MAX_PAGES_PER_DOC} limit)"))
            dest.unlink(missing_ok=True)
            continue

        if total_pages + pages > MAX_TOTAL_PAGES:
            skipped.append((url, "would exceed total page budget"))
            dest.unlink(missing_ok=True)
            continue

        accepted.append((dest.name, pages, url))
        total_pages += pages
        time.sleep(1)  # be polite to the server

    lines = [
        "# Corpus Fetch Report\n",
        f"Documents accepted: {len(accepted)}",
        f"Total pages: {total_pages}\n",
        "## Accepted\n",
    ]
    lines += [f"- `{n}` — {p} pages — {u}" for n, p, u in accepted]
    lines.append("\n## Skipped\n")
    lines += [f"- {u} — {r}" for u, r in skipped]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDone. {len(accepted)} documents, {total_pages} total pages.")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()