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
        scores = compute_composite_score(row_dict, sim, persona=persona)

        # Bandit adjustment
        bandit_boost = 0.0
        if bandit_scores:
            opp_id = row_dict["id"]
            bandit_boost = bandit_scores.get(opp_id, 0.5) - 0.5  # center at 0

        # Domain preference boost
        domain_boost = 0.0
        if domain_prefs:
            domain = row_dict.get("domain", "")
            domain_boost = domain_prefs.get(domain, 0.0) * 0.15

        final_score = scores["composite_score"] + 0.1 * bandit_boost + domain_boost
        final_score = max(0.0, min(1.0, final_score))

        row_dict.update(scores)
        row_dict["final_score"] = round(final_score, 4)
        row_dict["explanation"] = build_score_explanation(row_dict, scores)
        row_dict["suggested_actions"] = suggest_actions(row_dict)

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
    Relevance defined as: 1.0 if domain in persona_relevant_domains, else 0.1
    """
    n = min(500, len(all_ids))
    sample_ids = all_ids[:n]
    sample_embs = all_embeddings[:n]
    sample_df = df[df["id"].isin(sample_ids)].set_index("id")

    def relevance(row_id):
        try:
            domain = sample_df.loc[row_id, "domain"]
            return 1.0 if domain in persona_relevant_domains else 0.1
        except Exception:
            return 0.0

    # Method 1: Random baseline
    random_ids = np.random.permutation(sample_ids)[:k]
    random_scores = [relevance(i) for i in random_ids]

    # Method 2: Stars-only
    id_to_stars = {row["id"]: row.get("stars", 0) for _, row in sample_df.iterrows()}
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

    ideal = [1.0] * len(persona_relevant_domains) + [0.1] * (k - len(persona_relevant_domains))

    return {
        "Random Baseline": ndcg_at_k(random_scores, ideal, k),
        "Stars-Only Ranking": ndcg_at_k(stars_scores, ideal, k),
        "Embedding Similarity": ndcg_at_k(sim_scores, ideal, k),
        "Full Composite + Re-rank": ndcg_at_k(full_scores, ideal, k),
    }
