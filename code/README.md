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

> **Note:** The offline dataset (`data/opportunities.csv`, 10,500 records) and embeddings (`data/embeddings.npy`) are committed to the repo. Running `build_real_dataset.py` is optional and requires network/API access.

## Architecture

```
Multi-source ingestion (GitHub / Hacker News)
    ↓
Bloom Filter deduplication (Lecture 2)
    ↓
Sentence-BERT embedding — all-MiniLM-L6-v2, 384-dim (Lecture 5)
    ↓
FAISS IndexFlatIP (exact inner-product search, <2ms for 10k vectors)
IVFFlat used automatically if dataset exceeds 50k records
    ↓
Composite engagement scoring:
  relevance + community health + visibility + (1−effort) + trend
    ↓
Thompson Sampling re-ranking (Lecture 8)
    ↓
Diversity re-ranking (max 5 per domain in top results)
    ↓
Streamlit Dashboard
```

## 6 Core Capabilities

1. **Multi-Source Ingestion & On-Demand Refresh** — GitHub API + Hacker News Firebase API; Bloom filter dedup; Python queue-based streaming prototype
2. **Content Embedding & Similarity** — `all-MiniLM-L6-v2` (384-dim) + FAISS IndexFlatIP (exact search at ≤50k scale)
3. **Engagement Scoring & Ranking** — 5-component composite score (relevance, community, visibility, effort, trend); NDCG@10 evaluation benchmark
4. **Adaptive Learning** — Thompson Sampling bandit with 50-round simulation demo
5. **Batch Analytics** — domain health, trending repos, volume-over-time, rising opportunities (Pandas batch over offline snapshot)
6. **Dashboard & Brief** — ranked cards with "Why this?", suggested actions, CSV/JSON export

## Dataset

- `data/opportunities.csv` — 10,500 records across 15 technical domains
- Sources: **GitHub 7,500** · **Hacker News 3,000**
- All records labeled `data_source = "offline"` (pre-seeded snapshot graders can run without API access)
- Live-fetched records (via "Fetch Live Updates" button) labeled `data_source = "live"` and persisted to SQLite
- Collected via `build_real_dataset.py`; supplemented by live API on demand

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
│   ├── data_collector.py       # Live API collectors (GitHub + HN) + streaming ingester
│   ├── ai_actions.py           # Provider-agnostic AI suggestion layer
│   ├── db.py                   # SQLite utilities
│   ├── build_real_dataset.py   # Optional: re-fetch data from APIs (needs network)
│   ├── generate_offline_data.py # Offline dataset generator (GitHub + HN)
│   ├── requirements.txt
│   └── README.md
├── data/
│   ├── opportunities.csv       # Offline snapshot (10,500 records, pre-included)
│   ├── embeddings.npy          # Pre-computed embeddings (384-dim × 10,500)
│   └── embedding_ids.npy       # Embedding ID mapping
└── prompts.md
```

## Live Deployment

App URL: https://engageiq-bax423git-qianyingyang.streamlit.app
