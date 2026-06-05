# EngageIQ — Smart Engagement Opportunity Scorer

BAX-423 Big Data · Spring 2026 · Final Project  
**Yang, Alice** · UC Davis Graduate School of Management

---

## What It Does

EngageIQ discovers, scores, and ranks where a professional should invest their time online across GitHub, Reddit, and Hacker News. It covers **15 technical domains** and learns from your feedback to surface increasingly relevant opportunities.

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

The first run computes and caches embeddings (~20–40 seconds). Subsequent runs use the cache.

> **Note:** The offline dataset (`data/opportunities.csv`, 10,500 records) is already included. Re-fetching live data via `build_real_dataset.py` is optional and requires network/API access — it may overwrite the cache.

## Architecture

```
Multi-source ingestion (GitHub / Reddit / HN)
    ↓
Bloom Filter deduplication (Lecture 2)
    ↓
Sentence-BERT embedding — all-MiniLM-L6-v2, 384-dim (Lecture 5)
    ↓
FAISS IndexFlatIP (exact inner-product search, <2ms for 10k vectors)
IVFFlat used automatically if dataset exceeds 50k records
    ↓
Composite engagement scoring:
  0.40 × relevance + 0.30 × community + 0.20 × visibility + 0.10 × (1−effort)
    ↓
Thompson Sampling re-ranking (Lecture 8)
    ↓
Diversity re-ranking (max 5 per domain in top results)
    ↓
Streamlit Dashboard
```

## 6 Core Capabilities

1. **Multi-Source Ingestion & Streaming** — GitHub API, Reddit public JSON, Hacker News Firebase API; Bloom filter dedup; queue-based live ingestion simulation
2. **Content Embedding & Similarity** — `all-MiniLM-L6-v2` (384-dim) + FAISS IndexFlatIP (exact search at ≤50k scale)
3. **Engagement Scoring & Ranking** — 4-component composite score; NDCG@10 evaluation benchmark
4. **Adaptive Learning** — Thompson Sampling bandit with 50-round simulation demo
5. **Batch Analytics** — domain health, trending repos, volume-over-time, rising opportunities (Pandas batch over offline snapshot)
6. **Dashboard & Brief** — ranked cards with "Why this?", suggested actions, CSV/JSON export

## Dataset

- `data/opportunities.csv` — 10,500 records across 15 technical domains
- `data/engageiq.db` — SQLite offline snapshot (same 10,500 records, pre-seeded)
- Sources: GitHub (6,820), Reddit (2,757), Hacker News (923)
- 9,500 offline records + 1,000 live-labeled records
- Collected via `build_real_dataset.py`; supplemented by live API on demand

## Test Personas

| Persona | Role | Key Interests |
|---------|------|--------------|
| Sofia | ML Student / Portfolio Builder | Machine Learning, AI Research |
| David | DevOps Engineer / Niche Community | DevOps/K8s, Cloud APIs |
| Lina | Data Journalist / Trend Spotter | Trending Open-Source, AI Research |
| Raj | Startup Founder / Marketing | Developer Tools, B2B SaaS |

## Environment Variables (optional for live API)

```
GITHUB_TOKEN=ghp_...       # GitHub Personal Access Token (increases rate limit)
REDDIT_CLIENT_ID=...       # For authenticated Reddit API
REDDIT_CLIENT_SECRET=...   # For authenticated Reddit API
```

Without these, the app runs fully on the offline dataset. The "Fetch Live Updates" button uses unauthenticated API calls.

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
│   ├── data_collector.py       # Live API collectors + streaming ingester
│   ├── db.py                   # SQLite utilities
│   ├── build_real_dataset.py   # Optional: re-fetch data from APIs (needs network)
│   ├── requirements.txt
│   └── README.md
├── data/
│   ├── opportunities.csv       # Offline snapshot (10,500 records, pre-included)
│   ├── engageiq.db             # SQLite snapshot (same data, pre-seeded)
│   ├── embeddings.npy          # Cached embeddings (auto-generated on first run)
│   └── embedding_ids.npy       # Embedding ID mapping (auto-generated on first run)
└── prompts.md
```

## Live Deployment

App URL: *(to be added after deployment)*
