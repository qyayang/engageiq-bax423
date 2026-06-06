#!/usr/bin/env python3
"""
Fetch real HN stories from the Algolia HN Search API, organized by domain.
Replaces all synthetic/fallback HN records in opportunities.csv with
real news.ycombinator.com/item?id=<objectID> entries.

Strategy:
  - 5 keyword queries per domain, hitsPerPage=100 each
  - Deduplicate by objectID across queries and domains
  - Filter: must have a title + points >= 2
  - Keep top TARGET_PER_DOMAIN by (points + comments*0.5) per domain

Run from engageiq/ (project root):
    python3 code/fetch_real_hn.py

After this, rebuild embeddings:
    python3 code/rebuild_embeddings.py
"""
import csv
import json
import math
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "data" / "opportunities.csv"
ALGOLIA  = "https://hn.algolia.com/api/v1/search"

TARGET_PER_DOMAIN = 100   # real HN stories per domain
MIN_POINTS        = 2     # quality floor
HITS_PER_PAGE     = 100   # Algolia max reliable page size
DELAY             = 0.20  # seconds between API calls

# ── Domain → keyword queries ──────────────────────────────────────────────────
# 5 queries per domain; first hit that reaches TARGET stops further queries.
DOMAIN_QUERIES: dict[str, list[str]] = {
    "AI Research": [
        "LLM large language model",
        "artificial intelligence research paper",
        "GPT Claude Anthropic OpenAI",
        "AI safety alignment interpretability",
        "transformer neural network research",
    ],
    "Machine Learning": [
        "machine learning PyTorch TensorFlow",
        "deep learning training GPU",
        "ML model deployment inference",
        "scikit-learn gradient boosting XGBoost",
        "reinforcement learning reward",
    ],
    "DevOps/K8s": [
        "Kubernetes production cluster",
        "Terraform infrastructure as code",
        "CI/CD pipeline GitHub Actions",
        "Docker containerization deployment",
        "observability Prometheus Grafana",
    ],
    "Developer Tools": [
        "developer productivity tools",
        "CLI command line tool open source",
        "debugging profiling developer",
        "VS Code IDE extension",
        "API development REST GraphQL",
    ],
    "Cybersecurity": [
        "security vulnerability CVE exploit",
        "cybersecurity breach attack",
        "privacy data protection GDPR",
        "penetration testing red team",
        "cryptography encryption TLS",
    ],
    "B2B SaaS": [
        "SaaS startup launch product",
        "B2B software API business",
        "startup growth revenue",
        "developer tools company",
        "subscription billing API monetization",
    ],
    "Frontend (React/Web)": [
        "React TypeScript frontend",
        "JavaScript framework web",
        "Next.js Svelte Vue",
        "web performance CSS",
        "browser JavaScript V8 engine",
    ],
    "Python Data Eng": [
        "Python data engineering pipeline",
        "Apache Airflow Spark ETL",
        "pandas polars data processing",
        "dbt data transformation warehouse",
        "Kafka streaming data platform",
    ],
    "Mobile Dev (iOS/Flutter)": [
        "Flutter mobile app development",
        "iOS Swift Xcode",
        "Android Kotlin mobile",
        "React Native cross-platform",
        "mobile app performance",
    ],
    "GameDev (C++)": [
        "game development engine",
        "game programming C++ Rust",
        "Unity Unreal indie game",
        "graphics rendering shader GPU",
        "game engine architecture ECS",
    ],
    "Embedded Systems (C/RTOS)": [
        "embedded systems firmware",
        "RTOS microcontroller bare metal",
        "Rust embedded no-std",
        "IoT hardware ESP32 Arduino",
        "real-time operating system",
    ],
    "Blockchain": [
        "blockchain cryptocurrency",
        "Ethereum smart contract DeFi",
        "web3 decentralized application",
        "zero knowledge proof ZK",
        "crypto protocol layer2",
    ],
    "Cloud APIs": [
        "AWS cloud computing",
        "Google Cloud Platform GCP",
        "serverless functions lambda",
        "cloud architecture multi-cloud",
        "API gateway microservices",
    ],
    "Trending Open-Source": [
        "open source project release",
        "GitHub open source tool",
        "developer community software",
        "free software alternative",
        "open source show HN",
    ],
    "Beginner Coding": [
        "learn programming beginner",
        "coding tutorial first project",
        "software engineering career junior",
        "programming language learn",
        "computer science self-taught",
    ],
}


# ── Algolia fetch ─────────────────────────────────────────────────────────────

def _fetch_page(query: str, page: int = 0, timeout: int = 8) -> list[dict]:
    params = urllib.parse.urlencode({
        "query":       query,
        "tags":        "story",
        "hitsPerPage": HITS_PER_PAGE,
        "page":        page,
    })
    url = f"{ALGOLIA}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data.get("hits", [])
    except Exception as exc:
        print(f"    ⚠ fetch error ({exc}), skipping")
        return []


# ── Record conversion ─────────────────────────────────────────────────────────

def _recency_growth(created_str: str) -> float:
    """Higher growth_rate for more recent stories."""
    try:
        created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        days_old = (datetime.now(tz=timezone.utc) - created).days
        return round(max(0.0, 100.0 - days_old * 0.08), 2)
    except Exception:
        return 10.0


def hit_to_row(hit: dict, domain: str) -> dict | None:
    title    = (hit.get("title") or hit.get("story_title") or "").strip()
    obj_id   = str(hit.get("objectID", ""))
    points   = int(hit.get("points") or 0)
    comments = int(hit.get("num_comments") or 0)
    ext_url  = hit.get("url") or ""
    created  = hit.get("created_at", "")

    if not title or not obj_id:
        return None
    if points < MIN_POINTS:
        return None

    hn_url = f"https://news.ycombinator.com/item?id={obj_id}"

    description = f"{points} points, {comments} comments on Hacker News"

    raw_score    = points + comments * 0.5
    activity     = round(min(raw_score / 600.0, 1.0), 4)
    growth_rate  = _recency_growth(created)

    return {
        "id":               f"hn_{obj_id}",
        "source":           "hackernews",
        "record_type":      "hn_story",
        "data_source":      "offline",
        "title":            title,
        "description":      description,
        "url":              hn_url,
        "domain":           domain,
        "language":         "",
        "tags":             json.dumps([domain, "hackernews"]),
        "stars":            0,
        "forks":            0,
        "contributors":     0,
        "open_issues":      0,
        "good_first_issues":0,
        "comments":         comments,
        "upvotes":          points,
        "activity_score":   activity,
        "growth_rate":      growth_rate,
        "created_at":       created,
        "updated_at":       created,
        "url_type":         "hn_item",
        "url_valid":        "true",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return

    # ── Load existing GitHub records ──────────────────────────────────────────
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        all_rows   = list(reader)

    github_rows = [r for r in all_rows if r.get("source") == "github"]
    print(f"Keeping {len(github_rows)} GitHub records")
    print(f"Replacing {len(all_rows) - len(github_rows)} synthetic HN records\n")

    # ── Fetch real HN stories by domain ───────────────────────────────────────
    seen_ids: set[str] = set()   # global dedup across all domains
    hn_rows:  list[dict] = []

    for domain, queries in DOMAIN_QUERIES.items():
        domain_hits: list[dict] = []
        domain_ids:  set[str]   = set()

        for query in queries:
            if len(domain_hits) >= TARGET_PER_DOMAIN:
                break
            print(f"  [{domain}] query: {query!r}")
            hits = _fetch_page(query)
            time.sleep(DELAY)

            added = 0
            for hit in hits:
                obj_id = str(hit.get("objectID", ""))
                if not obj_id or obj_id in seen_ids or obj_id in domain_ids:
                    continue
                row = hit_to_row(hit, domain)
                if row is None:
                    continue
                domain_hits.append((hit.get("points", 0) + hit.get("num_comments", 0) * 0.5, row))
                domain_ids.add(obj_id)
                added += 1

            print(f"    → {added} new (domain total: {len(domain_hits)})")

        # Sort by score, take top TARGET_PER_DOMAIN
        domain_hits.sort(key=lambda x: x[0], reverse=True)
        for _, row in domain_hits[:TARGET_PER_DOMAIN]:
            oid = row["id"].replace("hn_", "")
            seen_ids.add(oid)
            hn_rows.append(row)

        print(f"  ✓ {domain}: {min(len(domain_hits), TARGET_PER_DOMAIN)} stories\n")

    # ── Combine and write ─────────────────────────────────────────────────────
    # Ensure all columns present
    for col in ("url_type", "url_valid"):
        if col not in fieldnames:
            fieldnames.append(col)

    final_rows = github_rows + hn_rows
    print(f"{'='*58}")
    print(f"  GitHub records : {len(github_rows):>5}")
    print(f"  Real HN stories: {len(hn_rows):>5}")
    print(f"  Total          : {len(final_rows):>5}")
    print(f"{'='*58}")

    # Verify ≥10,000
    if len(final_rows) < 10_000:
        shortage = 10_000 - len(final_rows)
        print(f"⚠  {shortage} short of 10,000 — consider adding more keyword queries")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\n  CSV updated: {CSV_PATH}")
    print("\n  ✅ Next step: python3 code/rebuild_embeddings.py")


if __name__ == "__main__":
    main()
