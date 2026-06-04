# Key AI Prompts — EngageIQ (BAX-423 Final)

## Prompt 1 — Architecture Design
**Prompt:** "I'm building EngageIQ for BAX-423 — a smart engagement opportunity scorer. The spec requires: multi-source ingestion from GitHub/Reddit/HN, Bloom filter dedup (Lecture 2), Sentence-BERT + FAISS retrieval (Lecture 5), multi-stage ranking with NDCG (Lecture 7), and Thompson Sampling bandit (Lecture 8). Help me design the full module structure and data flow before writing any code."

**Purpose:** Established the overall architecture and module breakdown (bloom_filter → embeddings → scoring → ranking → adaptive_learning) before coding, ensuring all 4 BAX-423 techniques had clear homes and data flowed cleanly between stages.

---

## Prompt 2 — Bloom Filter Implementation
**Prompt:** "Implement a Bloom filter in Python for streaming deduplication. Use SHA-256 based hashing, store bits in a bytearray (not a list) for memory efficiency, and include a stats() method that returns fill ratio and estimated false positive rate. Target 200k items at 1% FPR."

**Purpose:** Guided the memory-efficient bit-array implementation. Key refinement: switched from `list[bool]` to `bytearray` (8× memory reduction) after the initial draft used a plain list.

---

## Prompt 3 — FAISS IVF Index Choice
**Prompt:** "I have 10,000 documents embedded with all-MiniLM-L6-v2 (384-dim). Which FAISS index type should I use for ANN search? Explain the trade-off between IVFFlat, HNSW, and IndexFlatIP for this dataset size, and give the optimal nprobe setting."

**Purpose:** Confirmed IVFFlat with n_cells=sqrt(n)=100 and nprobe=25 as the right choice — flat index is adequate at 10k but IVF demonstrates the lecture technique properly. Output drove the build_faiss_index() implementation.

---

## Prompt 4 — Composite Engagement Scoring Formula
**Prompt:** "Design a composite engagement score for a content recommendation system. The score should combine: (1) semantic relevance to user interests, (2) community health signals (activity, responsiveness), (3) visibility potential (star count, contributor gap), (4) effort estimate (beginner-friendly signals). Justify the weights and explain how each component maps to a different engagement goal."

**Purpose:** Produced the 0.40/0.30/0.20/0.10 weight breakdown with rationale. Refined the effort component to use good-first-issue labels as a strong low-effort signal rather than just star count.

---

## Prompt 5 — Thompson Sampling vs Epsilon-Greedy
**Prompt:** "For the adaptive learning component, compare Thompson Sampling (Beta-Bernoulli bandit) versus epsilon-greedy bandit for a recommendation scenario where: feedback is sparse, we have 3 signal types (engage=1, bookmark=0.5, skip=0), and we want to demonstrate measurable improvement over 50 rounds. Which is better and why?"

**Purpose:** Confirmed Thompson Sampling as the better choice — natural handling of uncertainty, no epsilon hyperparameter to tune, and faster convergence on sparse feedback. This directly shaped the ThompsonBandit implementation and the fractional alpha update for bookmarks.

---

## Prompt 6 — Multi-Stage Ranking Pipeline
**Prompt:** "Implement a 3-stage ranking pipeline: (1) FAISS ANN retrieval of top-300 candidates, (2) composite scoring, (3) diversity re-ranking to prevent domain clustering. Include NDCG@10 evaluation that benchmarks 4 methods: random, stars-only, embedding-only, and full composite."

**Purpose:** Shaped the rank_candidates() and benchmark_ranking_methods() functions. The key insight from the refinement: diversity re-ranking should limit per-domain items in top-N, not globally, to preserve quality while ensuring breadth.

---

## Prompt 7 — Offline Dataset Generation Strategy
**Prompt:** "I need to generate 10,000+ realistic records across 15 technical domains (ML, DevOps/K8s, Blockchain, etc.) without making API calls. Each record needs rich enough text for Sentence-BERT to distinguish domains semantically. Design a template-based generator using log-normal distributions for stats and domain-specific vocabularies."

**Purpose:** Generated the DOMAIN_CFG structure with per-domain keywords, star distributions, and title templates. Refinement: increased template diversity to prevent embedding collapse (same template repeated too many times would cluster artificially).

---

## Prompt 8 — Streamlit App UI Design
**Prompt:** "Design a Streamlit dashboard for EngageIQ with: (1) user profile sidebar, (2) ranked opportunity cards with scores and 'Why this?' explanations, (3) analytics tab with Plotly charts, (4) learning tab showing Thompson Sampling convergence, (5) persona test panel, (6) export tab. Use a dark theme with GitHub-inspired colors. The layout should work well in a 10-minute demo."

**Purpose:** Produced the tab structure and card layout. Key modification: moved the benchmark panel to the Learning tab (rather than a separate tab) so it's co-located with the Thompson Sampling explanation for demo coherence.

---

## Prompt 9 — Streaming Deduplication Integration
**Prompt:** "Show how to integrate a Bloom filter into a streaming data ingestion pipeline. The producer fetches from GitHub/Reddit/HN APIs in a background thread, the consumer drains a Python queue. Track duplicates_blocked vs records_ingested for the demo. The Bloom filter should persist across fetch calls."

**Purpose:** Shaped the StreamingIngester class and its producer/consumer pattern. This ensures the Bloom filter dedup is visible in the UI (sidebar shows items added / duplicates blocked) and maps clearly to the Kafka streaming architecture described in Lecture 3.

---

## Prompt 10 — NDCG Benchmark Explanation for Brief
**Prompt:** "Write a 150-word explanation of why NDCG@10 is the right evaluation metric for this recommendation system, comparing it to Precision@K and MRR. Then show what NDCG=0.72 vs NDCG=0.31 means in practical terms for a user looking at the top-10 ranked results."

**Purpose:** Drafted the metric discussion in brief.pdf. Key output: framing NDCG as the natural metric because our relevance is graded (engage > bookmark > skip), not binary — aligning with the listwise ranking discussion from Lecture 7.
