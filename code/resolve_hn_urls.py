#!/usr/bin/env python3
"""
Batch resolver: maps synthetic HN story titles to real HN item IDs
via the Algolia HN search API.

Key insight: 2,824 HN records share only ~75 unique normalized titles.
We resolve each unique title once, then apply the result to all records.

Strategy v2 (improved):
  - Multi-query per title: em-dash truncation, year stripping, Ask/Show prefix removal
  - Token overlap (Jaccard on 4+ char words) alongside SequenceMatcher — accepts
    topically related matches where exact phrasing differs
  - Threshold 0.30 — more recall, still filters random mismatches

url_type values:
  hn_item            — resolved to news.ycombinator.com/item?id=<objectID>
  hn_search_fallback — unresolved; Algolia search URL kept
  github_issue       — direct GitHub issue URL
  github_repo        — direct GitHub repo URL

Run from engageiq/code/:
    python3 resolve_hn_urls.py
"""
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

CSV_PATH  = Path(__file__).parent.parent / "data" / "opportunities.csv"
ALGOLIA   = "https://hn.algolia.com/api/v1/search"
DELAY     = 0.10   # seconds between API calls

# Dual thresholds (EITHER criterion triggers a match):
#   TOK5_THRESH: Jaccard on ≥5-char words — robust topic/concept overlap
#   SEQ_THRESH:  SequenceMatcher on prefix-stripped strings — near-duplicates
#
# Stripping "Ask HN:" / "Show HN:" before seq comparison prevents false
# positives from shared template prefixes like "Ask HN: What's the best X vs
# Ask HN: What's the best Y" which would otherwise score ~0.65+ on seq.
TOK5_THRESH = 0.30   # Jaccard on ≥5-char words
SEQ_THRESH  = 0.65   # SequenceMatcher after stripping HN prefix


# ── Similarity helpers ────────────────────────────────────────────────────────

_HN_PREFIX = re.compile(r'^(Ask|Show) HN:\s*', re.IGNORECASE)


def _strip_hn(s: str) -> str:
    return _HN_PREFIX.sub("", s).strip()


def _seq_sim(a: str, b: str) -> float:
    """SequenceMatcher on prefix-stripped strings to avoid template bias."""
    return SequenceMatcher(None, _strip_hn(a).lower(), _strip_hn(b).lower()).ratio()


def _tok5(a: str, b: str) -> float:
    """Jaccard on ≥5-char words — topical overlap, filters short template words."""
    wa = set(re.findall(r'\b\w{5,}\b', a.lower()))
    wb = set(re.findall(r'\b\w{5,}\b', b.lower()))
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _accept(a: str, b: str) -> bool:
    """True if either topical word overlap or near-duplicate string match."""
    return _tok5(a, b) >= TOK5_THRESH or _seq_sim(a, b) >= SEQ_THRESH


def _score(a: str, b: str) -> float:
    """Score for picking best Algolia hit (higher = better)."""
    return max(_tok5(a, b), _seq_sim(a, b))


# ── Query building ────────────────────────────────────────────────────────────

def _normalize(raw: str) -> str:
    """Canonical form used for deduplication and as query base."""
    t = re.sub(r"\s*\(\d+\)\s*$", "", raw).strip()   # strip (N) suffix
    t = re.sub(r"\bin\s+20\d\d\b", "", t).strip()    # strip year ref
    t = re.sub(r"\s+", " ", t)
    return t


def _smart_queries(norm: str) -> list[str]:
    """
    Generate 2-3 query variants for better Algolia recall.
    Tries: (1) em-dash truncation + year stripped, (2) Ask/Show HN: prefix removed,
           (3) first 5 content words of the question.
    """
    queries: list[str] = []

    # Variant 1: strip em-dash suffix (keep left side), strip year
    base = norm.split("—")[0].strip()
    base_ny = re.sub(r"\bin\s+20\d\d\b", "", base).strip()
    base_ny = re.sub(r"\s+", " ", base_ny).strip()
    queries.append(base_ny[:65])

    # Variant 2: strip Ask/Show HN: prefix
    clean = re.sub(r"^(Ask|Show) HN:\s*", "", base_ny).strip()
    if clean and clean != base_ny and len(clean) >= 8:
        queries.append(clean[:65])

    # Variant 3: first 5 words of the question part
    words = clean.split()
    if len(words) > 5:
        queries.append(" ".join(words[:5]))

    return list(dict.fromkeys(q for q in queries if q))


# ── Resolver ──────────────────────────────────────────────────────────────────

def _resolve_title(norm_title: str, timeout: int = 6) -> tuple[str | None, str | None]:
    """
    Try each query variant against Algolia.
    Return (objectID, matched_title) on first hit >= THRESHOLD, else (None, None).
    """
    for query in _smart_queries(norm_title):
        params = urllib.parse.urlencode({"query": query, "hitsPerPage": 5})
        url    = f"{ALGOLIA}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read())
            hits = [h for h in data.get("hits", []) if "story" in h.get("_tags", [])]
            if hits:
                best = max(hits, key=lambda h: _score(norm_title, h.get("title", "")))
                if _accept(norm_title, best.get("title", "")):
                    return str(best["objectID"]), best.get("title", "")
        except Exception:
            pass
        time.sleep(DELAY)
    return None, None


# ── GitHub classifier ─────────────────────────────────────────────────────────

def _classify_github(url: str) -> str:
    if "github.com/user/" in url or "github.com/search" in url:
        return "invalid"
    if "/issues/" in url:
        return "github_issue"
    if "github.com/" in url:
        return "github_repo"
    return "unknown"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows       = list(reader)

    for col in ("url_type", "url_valid"):
        if col not in fieldnames:
            fieldnames.append(col)

    hn_rows = [r for r in rows if r.get("source") == "hackernews"]
    gh_rows = [r for r in rows if r.get("source") == "github"]
    print(f"Records: {len(rows)}  |  HN: {len(hn_rows)}  |  GitHub: {len(gh_rows)}")

    # ── Step 1: group HN records by normalized title ──────────────────────────
    norm_to_rows: dict[str, list] = defaultdict(list)
    for r in hn_rows:
        norm_to_rows[_normalize(r.get("title", ""))].append(r)

    unique_titles = sorted(norm_to_rows)
    print(f"Unique HN normalized titles: {len(unique_titles)}")
    print(f"Resolving via Algolia (tok5>={TOK5_THRESH} or seq>={SEQ_THRESH}, multi-query) …\n")

    # ── Step 2: resolve each unique title once ────────────────────────────────
    title_to_id:    dict[str, str] = {}
    title_to_match: dict[str, str] = {}

    resolved_titles = 0
    fallback_titles = 0

    for i, norm in enumerate(unique_titles):
        obj_id, matched = _resolve_title(norm)
        if obj_id:
            title_to_id[norm]    = obj_id
            title_to_match[norm] = matched
            resolved_titles += 1
            print(f"  ✓ [{i+1}/{len(unique_titles)}] {norm[:55]}")
            print(f"         → https://news.ycombinator.com/item?id={obj_id}")
            print(f"           matched: {matched[:60]}")
        else:
            fallback_titles += 1
            if i % 5 == 0 or fallback_titles <= 5:
                print(f"  ✗ [{i+1}/{len(unique_titles)}] {norm[:55]}")

    print(f"\nTitle resolve: {resolved_titles}/{len(unique_titles)} titles resolved  "
          f"({100*resolved_titles/max(len(unique_titles),1):.0f}%)\n")

    # ── Step 3: apply results to all rows ────────────────────────────────────
    resolved_records = 0
    fallback_records = 0

    for row in rows:
        src = row.get("source", "")

        if src == "github":
            ut = _classify_github(row.get("url", ""))
            row["url_type"]  = ut
            row["url_valid"] = "true" if ut in ("github_issue", "github_repo") else "false"
            continue

        if src != "hackernews":
            row["url_type"]  = "unknown"
            row["url_valid"] = "false"
            continue

        norm = _normalize(row.get("title", ""))
        if norm in title_to_id:
            obj_id = title_to_id[norm]
            row["url"]       = f"https://news.ycombinator.com/item?id={obj_id}"
            row["url_type"]  = "hn_item"
            row["url_valid"] = "true"
            resolved_records += 1
        else:
            row["url_type"]  = "hn_search_fallback"
            row["url_valid"] = "false"
            fallback_records += 1

    # ── Step 4: write CSV ─────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    total_hn    = resolved_records + fallback_records
    resolve_pct = resolved_records / max(total_hn, 1) * 100
    print(f"{'='*58}")
    print(f"  HN resolved  : {resolved_records:>5}  ({resolve_pct:.1f}%  of {total_hn} HN records)")
    print(f"  HN fallback  : {fallback_records:>5}")
    print(f"  GitHub typed : {len(gh_rows):>5}  (github_issue / github_repo)")
    print(f"  CSV updated  : {CSV_PATH}")
    print(f"{'='*58}")
    print("\nEmbeddings do NOT need regeneration (title/description unchanged).")


if __name__ == "__main__":
    main()
