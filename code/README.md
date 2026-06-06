# EngageIQ — Smart Engagement Opportunity Scorer

BAX-423 Big Data · Spring 2026 · Final Project  
**Yang, Alice** · UC Davis Graduate School of Management

---

## What It Does

EngageIQ discovers, scores, and ranks where a professional should invest their time online across **GitHub and Hacker News**. It covers **15 technical domains** and learns from your feedback to surface increasingly relevant opportunities.

## BAX-423 Techniques Integrated

| Technique | Lecture | Implementation |
|-----------|---------|----------------|
| **Bloom Filter** (sketching) | Lecture 2 | Streaming deduplication — `bloom_filter.py` |
| **Sentence-BERT + FAISS** | Lecture 5 | ANN retrieval — `embeddings.py` |
| **Multi-stage Ranking** (NDCG) | Lecture 7 | Candidate gen → scoring → re-rank — `ranking.py` |
| **Thompson Sampling** (RL bandit) | Lecture 8 | Adaptive learning — `adaptive_learning.py` |

## Quick Start

```bash
# 1. Install dependencies (from the engageiq/ root or code/ directory)
pip install -r requirements.txt

# 2. Launch the app — offline dataset is pre-loaded, no data generation needed
cd code
streamlit run app.py
```

The first run loads the pre-computed embedding cache (~1 second). No re-computation needed.

> **Note:** The offline dataset (`data/opportunities.csv`, 10,046 records) and embeddings (`data/embeddings.npy`) are committed to the repo. Running `build_real_dataset.py` is optional and requires network/API access.

## Architecture

```
Multi-source ingestion (GitHub / Hacker News)
    ↓
Bloom Filter deduplication (Lecture 2)
    ↓
Intent Inference — role + interest keywords → intent label
  (mobile_contribution / data_engineering / startup_growth /
   security_review / contribution / trend_spotting / community_engagement)
    ↓
Adaptive Query Expansion — intent-specific keyword augmentation for FAISS recall
  (mobile adds "iOS Android Flutter Swift Kotlin Dart…";
   data engineering adds "ETL pipeline Airflow dbt Spark…")
    ↓
FAISS IndexFlatIP ANN retrieval + Domain-Specific Candidate Injection (Lecture 5)
  (injects Mobile Dev / Python Data Eng records FAISS underweights)
    ↓
Composite engagement scoring with intent-matched persona weights:
  relevance + community health + visibility + (1−effort) + trend  (Lecture 7)
    ↓
Intent-aware reranking (GFI boost / domain boost / source caps per intent)
    ↓
Per-domain diversity cap (max 4 per domain in top results)
    ↓
Thompson Sampling re-ranking (Lecture 8)
    ↓
Streamlit Dashboard
```

## 7 Core Capabilities

1. **Multi-Source Ingestion & On-Demand Refresh** — GitHub API + Hacker News Firebase API; Bloom filter dedup; Python queue-based streaming prototype
2. **Content Embedding & Similarity** — `all-MiniLM-L6-v2` (384-dim) + FAISS IndexFlatIP (exact search at ≤50k scale)
3. **Adaptive Intent Inference** — 7-intent classifier (mobile, data engineering, security, startup growth, contribution, trend spotting, community engagement) inferred from role + interest keywords; drives adaptive query, candidate injection, and reranking — shared by main flow and persona test panel
4. **Engagement Scoring & Ranking** — 5-component composite score (relevance, community, visibility, effort, trend); intent-matched persona weights; NDCG@10 evaluation benchmark
5. **Adaptive Learning** — Thompson Sampling bandit with 50-round simulation demo
6. **Batch Analytics** — domain health, trending repos, volume-over-time, rising opportunities (Pandas batch over offline snapshot)
7. **Dashboard & Brief** — ranked cards with "Why this?", suggested actions, CSV/JSON export

## Dataset

- `data/opportunities.csv` — 10,046 records across 15 technical domains
- Sources: **GitHub 8,588** (real API-derived issues + repos; direct links) · **Hacker News 1,458** (real Algolia/HN API stories)
- All records labeled `data_source = "offline"` (pre-seeded snapshot; graders can run without API access)
- GitHub records: real API-fetched — direct `github.com/{owner}/{repo}/issues/{N}` and `github.com/{owner}/{repo}` URLs
- HN records: real Algolia/HN API stories fetched by domain keyword queries — direct `news.ycombinator.com/item?id=<objectID>` URLs; no `hn_search_fallback` records in the submitted dataset
- `url_type` field in CSV: `github_issue` / `github_repo` / `hn_item` (0 `hn_search_fallback`)
- Live-fetched records (via "Fetch Live Updates" button) labeled `data_source = "live"` and persisted to SQLite
- Collected via `build_real_dataset.py` and `fetch_real_hn.py`; supplemented by live API on demand

## Why GitHub + Hacker News?

- **GitHub** — contribution opportunities: repos with good-first-issues, open issues, and community health signals
- **Hacker News** — trend discovery: discussion threads that surface rising technologies and topics before they go mainstream
- Together they cover both *actionability* (GitHub PRs/issues) and *trend awareness* (HN stories)

## Test Personas

| Persona | Role | Key Interests |
|---------|------|--------------|
| Sofia | ML Student / Portfolio Builder | Machine Learning, AI Research |
| David | DevOps Engineer / Niche Community | DevOps/K8s, Cloud APIs |
| Lina | Data Journalist / Trend Spotter | Trending Open-Source, AI Research |
| Raj | Startup Founder / Marketing | Developer Tools, B2B SaaS |

## Adaptive Intent Layer & Robustness

The same intent inference, query expansion, candidate injection, and intent-aware reranking layer powers both the **main Action Queue / Full Ranked List** and the **Persona Test Panel**. Hidden personas get the same quality as the 4 graded ones — not a separate test-only path.

**Stress-tested against 11 unseen-adjacent hidden roles (all 11/11 PASS):**

| Hidden Persona | Inferred Intent | Result |
|----------------|-----------------|--------|
| Security Researcher | security_review | ✅ PASS |
| Climate Tech Founder | startup_growth | ✅ PASS |
| Beginner Developer | contribution | ✅ PASS |
| Open Source Maintainer | community_engagement | ✅ PASS |
| Product Manager | startup_growth | ✅ PASS |
| Mobile Developer | mobile_contribution | ✅ PASS |
| Game Developer | generic | ✅ PASS |
| Data Engineer | contribution | ✅ PASS |
| Academic ML Researcher | generic | ✅ PASS |
| Education Creator | trend_spotting | ✅ PASS |
| Privacy Researcher | security_review | ✅ PASS |

Pass criteria: domain_match ≥4/10 · primary_interest_match ≥1/10 · real_url_count ≥4/10 (direct GitHub/HN links only; Algolia fallbacks excluded) · domain_diversity ≥2 · interest_kw_hit ≥2 · source_fit ≥6/10 · neg_filter = 0

## Environment Variables (optional for live API)

```
GITHUB_TOKEN=ghp_...       # GitHub Personal Access Token (increases rate limit)
DEEPSEEK_API_KEY=...       # Optional: AI-generated engagement suggestions
LLM_PROVIDER=deepseek      # Optional: openai | anthropic | gemini | deepseek | ollama
```

Without these, the app runs fully on the offline dataset. The "Fetch Live Updates" button fetches from GitHub and Hacker News APIs.

## File Structure

```
engageiq/
├── code/
│   ├── app.py                  # Streamlit dashboard (main entry)
│   ├── bloom_filter.py         # Bloom filter — BAX-423 Lecture 2
│   ├── embeddings.py           # Sentence-BERT + FAISS — BAX-423 Lecture 5
│   ├── scoring.py              # Composite engagement scoring
│   ├── ranking.py              # Multi-stage ranking — BAX-423 Lecture 7
│   ├── adaptive_learning.py    # Thompson Sampling — BAX-423 Lecture 8
│   ├── analytics.py            # Batch analytics & trend detection
│   ├── ai_actions.py           # Provider-agnostic AI suggestion layer
│   ├── data_collector.py       # Live API collectors (GitHub + HN) + streaming ingester
│   ├── db.py                   # SQLite utilities
│   ├── build_real_dataset.py   # Optional: re-fetch data from APIs (needs network)
│   ├── fetch_real_hn.py        # Fetch real HN stories via Algolia API (replaces synthetic HN)
│   ├── rebuild_embeddings.py   # Rebuild embeddings.npy after CSV changes
│   ├── generate_offline_data.py # Offline dataset generator (GitHub + HN)
│   ├── requirements.txt
│   └── README.md
├── data/
│   ├── opportunities.csv       # Offline snapshot (11,412 records, pre-included)
│   ├── embeddings.npy          # Pre-computed embeddings (384-dim × 11,412)
│   └── embedding_ids.npy       # Embedding ID mapping
└── prompts.md
```

## Live Deployment

App URL: https://engageiq-bax423git-qianyingyang.streamlit.app
