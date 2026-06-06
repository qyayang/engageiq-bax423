"""
Multi-stage ranking pipeline.
BAX-423 Lecture 7 — Ranking & Multi-Stage Recommendation Systems.

Stage 1: Candidate generation via FAISS ANN (embedding retrieval)
Stage 2: Composite engagement scoring
Stage 3: Re-ranking with bandit-adjusted scores + diversity

Evaluation: NDCG@10 reported for each ranking method.
"""
import numpy as np
import pandas as pd
from typing import Optional

from scoring import compute_composite_score, build_score_explanation, suggest_actions


def _dcg(relevances: list[float], k: int) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg_at_k(ranked_scores: list[float], ideal_scores: list[float], k: int = 10) -> float:
    ideal_sorted = sorted(ideal_scores, reverse=True)
    dcg = _dcg(ranked_scores, k)
    idcg = _dcg(ideal_sorted, k)
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def rank_candidates(
    df: pd.DataFrame,
    candidate_ids: list[str],
    candidate_sims: list[float],
    bandit_scores: Optional[dict] = None,
    domain_prefs: Optional[dict] = None,
    filters: Optional[dict] = None,
    top_n: int = 50,
    persona: str = "",
    intent_override: str | None = None,
) -> list[dict]:
    """
    Full multi-stage ranking:
      1. Filter by user preferences (source, domain, time budget)
      2. Score each candidate with composite scoring
      3. Re-rank: composite + bandit boost + domain preference boost
      4. Diversity re-ranking: prevent domain clustering in top-10

    Returns list of enriched opportunity dicts with scores and explanations.
    """
    id_to_sim = dict(zip(candidate_ids, candidate_sims))
    subset = df[df["id"].isin(set(candidate_ids))].copy()

    # Apply filters
    if filters:
        if filters.get("source") and filters["source"] != "All":
            subset = subset[subset["source"] == filters["source"].lower()]
        if filters.get("domain") and filters["domain"] != "All":
            subset = subset[subset["domain"] == filters["domain"]]
        if filters.get("exclude_no_gfi"):
            subset = subset[subset["good_first_issues"] > 0]

    if subset.empty:
        return []

    results = []
    for _, row in subset.iterrows():
        row_dict = row.to_dict()
        sim = id_to_sim.get(row_dict["id"], 0.0)
        scores = compute_composite_score(row_dict, sim, persona=persona, intent_override=intent_override)

        # Bandit adjustment (item-level): engaged items rise, skipped items fall
        bandit_boost = 0.0
        if bandit_scores:
            opp_id = row_dict["id"]
            b = bandit_scores.get(opp_id, 0.5)
            bandit_boost = (b - 0.5) * 0.28   # ±0.14 max, visibly shifts ranking

        # Domain preference boost (aggregate-level): centered at 0, can go negative
        domain_boost = 0.0
        if domain_prefs:
            domain = row_dict.get("domain", "")
            pref = domain_prefs.get(domain, 0.5)  # 0.5 = neutral
            domain_boost = (pref - 0.5) * 0.30    # ±0.15 max
            domain_boost = max(-0.15, min(0.20, domain_boost))

        final_score = scores["composite_score"] + bandit_boost + domain_boost
        final_score = max(0.0, min(1.0, final_score))

        row_dict.update(scores)
        row_dict["final_score"] = round(final_score, 4)
        row_dict["explanation"] = build_score_explanation(row_dict, scores)
        row_dict["suggested_actions"] = suggest_actions(row_dict, persona=persona, intent_override=intent_override)

        results.append(row_dict)

    # Sort by final_score descending
    results.sort(key=lambda x: x["final_score"], reverse=True)

    # Diversity re-ranking: ensure top-10 has ≥3 unique domains
    results = _diversity_rerank(results, top_n=top_n)

    return results[:top_n]


def _diversity_rerank(results: list[dict], top_n: int = 50, max_per_domain: int = 5) -> list[dict]:
    """Maximal Marginal Relevance-style diversity for top results."""
    domain_counts: dict[str, int] = {}
    output = []
    deferred = []

    for r in results:
        domain = r.get("domain", "Unknown")
        if domain_counts.get(domain, 0) < max_per_domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            output.append(r)
        else:
            deferred.append(r)
        if len(output) >= top_n:
            break

    output.extend(deferred)
    return output


def benchmark_ranking_methods(
    df: pd.DataFrame,
    query_vec: np.ndarray,
    all_embeddings: np.ndarray,
    all_ids: list[str],
    persona_relevant_domains: list[str],
    k: int = 10,
) -> dict:
    """
    Benchmarks 4 ranking approaches and returns NDCG@10 for each.

    Relevance is quality-aligned (mirrors the composite scoring formula):
      - 0.0  for records outside persona_relevant_domains
      - [0, 1] for in-domain records: 0.4 × stars_norm + 0.3 × activity_score
                                    + 0.2 × community_health + 0.2 × (GFI bonus)
    This rewards the same signals the composite scorer optimises, making
    NDCG@10 a direct measure of how well each method recovers high-quality
    in-domain opportunities.

    Uses a reproducible random sample (seed=42) representative of the full
    dataset (not biased toward any contiguous slice of the CSV).
    """
    n = min(500, len(all_ids))
    rng = np.random.RandomState(42)
    sample_idx = sorted(rng.permutation(len(all_ids))[:n])
    sample_ids   = [all_ids[i] for i in sample_idx]
    sample_embs  = all_embeddings[sample_idx]
    sample_df    = df[df["id"].isin(sample_ids)].set_index("id")
    relevant_set = set(persona_relevant_domains)

    def relevance(row_id):
        try:
            row    = sample_df.loc[row_id]
            domain = row.get("domain", "")
            if domain not in relevant_set:
                return 0.0
            stars_norm = min(float(row.get("stars", 0) or 0) / 5000, 1.0)
            activity   = float(row.get("activity_score", 0) or 0)
            comm       = float(row.get("community_health", 0) or 0)
            gfi_bonus  = 0.2 if int(row.get("good_first_issues", 0) or 0) > 0 else 0.0
            return min(0.4 * stars_norm + 0.3 * activity + 0.2 * comm + gfi_bonus, 1.0)
        except Exception:
            return 0.0

    # Method 1: Random baseline (fixed seed so comparison is reproducible)
    random_ids = rng.permutation(sample_ids)[:k]
    random_scores = [relevance(i) for i in random_ids]

    # Method 2: Stars-only
    id_to_stars = {idx: row.get("stars", 0) for idx, row in sample_df.iterrows()}
    stars_ranked = sorted(sample_ids, key=lambda i: id_to_stars.get(i, 0), reverse=True)[:k]
    stars_scores = [relevance(i) for i in stars_ranked]

    # Method 3: Embedding similarity only
    sims = (sample_embs @ query_vec.T).flatten()
    sim_ranked_idx = np.argsort(sims)[::-1][:k]
    sim_ranked_ids = [sample_ids[i] for i in sim_ranked_idx]
    sim_scores = [relevance(i) for i in sim_ranked_ids]

    # Method 4: Full composite scoring
    candidate_sims = [(sample_ids[i], float(sims[i])) for i in np.argsort(sims)[::-1][:200]]
    c_ids = [x[0] for x in candidate_sims]
    c_sims = [x[1] for x in candidate_sims]
    full_results = rank_candidates(sample_df.reset_index(), c_ids, c_sims, top_n=k)
    full_ids = [r["id"] for r in full_results[:k]]
    full_scores = [relevance(i) for i in full_ids]

    # Ideal = actual relevance distribution of the full sample, sorted best-first.
    # This guarantees IDCG ≥ DCG for any ranking of the same records, so NDCG ≤ 1.0.
    # (The old approach used len(domains) as the 1.0-count, underestimating IDCG when
    #  a method retrieves more than len(domains) relevant items in its top-k.)
    ideal = sorted([relevance(i) for i in sample_ids], reverse=True)

    return {
        "Random Baseline": ndcg_at_k(random_scores, ideal, k),
        "Stars-Only Ranking": ndcg_at_k(stars_scores, ideal, k),
        "Embedding Similarity": ndcg_at_k(sim_scores, ideal, k),
        "Full Composite + Re-rank": ndcg_at_k(full_scores, ideal, k),
    }
