"""
Composite engagement scoring.
Implements the 4-component score used in the multi-stage ranking pipeline.
"""
import math
import numpy as np
import pandas as pd


# Default weights for composite engagement score
W_RELEVANCE = 0.40
W_COMMUNITY = 0.30
W_VISIBILITY = 0.20
W_EFFORT_INV = 0.10

# Persona-specific scoring strategies (BAX-423: different users have different goals)
PERSONA_WEIGHTS = {
    "ML Student": {
        "relevance": 0.45, "community": 0.20, "visibility": 0.20, "effort_inv": 0.15,
        "gfi_bonus": 0.12,   # strong boost for good-first-issues
    },
    "DevOps Engineer": {
        "relevance": 0.40, "community": 0.38, "visibility": 0.15, "effort_inv": 0.07,
        "gfi_bonus": 0.0,    # expert — doesn't need beginner issues
    },
    "Data Journalist": {
        "relevance": 0.25, "community": 0.20, "visibility": 0.35, "effort_inv": 0.05,
        "gfi_bonus": 0.0,    # prioritises recency and velocity over effort
    },
    "Startup Founder": {
        "relevance": 0.45, "community": 0.28, "visibility": 0.18, "effort_inv": 0.09,
        "gfi_bonus": 0.0,
    },
    "Data Engineer": {
        "relevance": 0.42, "community": 0.30, "visibility": 0.18, "effort_inv": 0.10,
        "gfi_bonus": 0.05,
    },
}


def _safe_log_norm(x: float, scale: float = 1.0) -> float:
    return min(math.log1p(max(x, 0)) / math.log1p(scale), 1.0)


def compute_community_health(row: dict) -> float:
    """Normalized signal reflecting community activity and responsiveness."""
    source = row.get("source", "")
    if source == "github":
        contrib = _safe_log_norm(row.get("contributors", 0), 200)
        issues = _safe_log_norm(row.get("open_issues", 0), 500)
        growth = _safe_log_norm(row.get("growth_rate", 0), 200)
        return round(0.40 * contrib + 0.30 * issues + 0.30 * growth, 4)
    elif source == "reddit":
        comments = _safe_log_norm(row.get("comments", 0), 300)
        upvotes = _safe_log_norm(row.get("upvotes", 0), 5000)
        return round(0.50 * comments + 0.50 * upvotes, 4)
    elif source == "hackernews":
        comments = _safe_log_norm(row.get("comments", 0), 200)
        upvotes = _safe_log_norm(row.get("upvotes", 0), 1000)
        return round(0.40 * comments + 0.60 * upvotes, 4)
    return 0.0


def compute_visibility_potential(row: dict) -> float:
    """
    Higher stars with fewer contributors = more opportunity to stand out.
    For Reddit/HN: upvote count drives visibility.
    """
    source = row.get("source", "")
    if source == "github":
        stars_norm = _safe_log_norm(row.get("stars", 0), 50000)
        contrib = max(row.get("contributors", 1), 1)
        stars = max(row.get("stars", 0), 0)
        gap = min(stars / contrib / 1000, 1.0)  # contributor gap score
        return round(0.60 * stars_norm + 0.40 * gap, 4)
    else:
        upvotes = _safe_log_norm(row.get("upvotes", 0) + row.get("comments", 0), 5000)
        return round(upvotes, 4)


def compute_effort_score(row: dict) -> float:
    """Estimate time/effort required — lower is better for the composite."""
    source = row.get("source", "")
    if source == "github":
        gfi = row.get("good_first_issues", 0)
        if gfi > 0:
            return 0.2
        stars = row.get("stars", 0)
        if stars < 500:
            return 0.3
        elif stars < 5000:
            return 0.5
        else:
            return 0.7
    else:
        words = len(str(row.get("description", "")).split())
        return min(words / 300, 0.8)


def compute_composite_score(row: dict, relevance: float, persona: str = "") -> dict:
    """
    Returns the full score breakdown for a single opportunity.
    relevance: cosine similarity from embedding retrieval (0–1).
    """
    community = compute_community_health(row)
    visibility = compute_visibility_potential(row)
    effort = compute_effort_score(row)
    effort_inv = 1.0 - effort

    pw = PERSONA_WEIGHTS.get(persona, {})
    w_rel = pw.get("relevance", W_RELEVANCE)
    w_com = pw.get("community", W_COMMUNITY)
    w_vis = pw.get("visibility", W_VISIBILITY)
    w_eff = pw.get("effort_inv", W_EFFORT_INV)

    composite = w_rel * relevance + w_com * community + w_vis * visibility + w_eff * effort_inv

    # Persona-specific bonus: ML students get a GFI boost
    gfi_bonus = pw.get("gfi_bonus", 0.0)
    if gfi_bonus > 0 and row.get("good_first_issues", 0) > 0:
        composite += gfi_bonus

    # Data journalist: recency boost (recent records score higher)
    if persona == "Data Journalist":
        growth = row.get("growth_rate", 0)
        composite += min(growth / 500, 0.10)

    composite = max(0.0, min(1.0, composite))

    return {
        "composite_score": round(composite, 4),
        "relevance_score": round(relevance, 4),
        "community_health": round(community, 4),
        "visibility_score": round(visibility, 4),
        "effort_score": round(effort, 4),
    }


def build_score_explanation(row: dict, scores: dict) -> str:
    """Human-readable explanation for 'Why this?' feature."""
    parts = []
    rel = scores["relevance_score"]
    comm = scores["community_health"]
    vis = scores["visibility_score"]
    eff = scores["effort_score"]

    parts.append(f"**Relevance to your interests:** {rel:.0%}")

    source = row.get("source", "")
    if source == "github":
        gfi = row.get("good_first_issues", 0)
        contrib = row.get("contributors", 0)
        stars = row.get("stars", 0)
        if gfi > 0:
            parts.append(f"**{gfi} good-first-issues** available — beginner-friendly")
        parts.append(f"**Community:** {contrib} contributors, {stars:,} stars — health score {comm:.0%}")
        parts.append(f"**Visibility:** Contributor-to-star ratio favors newcomers ({vis:.0%})")
        if eff < 0.35:
            parts.append("**Effort:** Low — labeled as beginner-friendly")
        elif eff < 0.6:
            parts.append(f"**Effort:** Moderate — mid-size project")
        else:
            parts.append(f"**Effort:** Higher — large, active codebase")
    elif source == "reddit":
        sub = row.get("tags", "")
        comments = row.get("comments", 0)
        upvotes = row.get("upvotes", 0)
        parts.append(f"**Thread activity:** {comments} comments, {upvotes} upvotes — {comm:.0%} community health")
        parts.append(f"**Visibility:** Your comment reaches {vis:.0%} of potential audience")
        parts.append(f"**Effort:** {'Low' if eff < 0.4 else 'Moderate'} — conversational thread")
    else:
        comments = row.get("comments", 0)
        score = row.get("upvotes", 0)
        parts.append(f"**HN score:** {score} points, {comments} comments — {comm:.0%} engagement health")
        parts.append(f"**Visibility:** High-quality comment surfaces in {vis:.0%} of reader feeds")

    parts.append(f"\n**Overall engagement score: {scores['composite_score']:.0%}**")
    return "\n\n".join(parts)


def suggest_actions(row: dict) -> list[str]:
    """Rule-based engagement suggestions (LLM-style quality without API call)."""
    source = row.get("source", "")
    title = row.get("title", "")
    domain = row.get("domain", "")
    actions = []

    if source == "github":
        gfi = row.get("good_first_issues", 0)
        stars = row.get("stars", 0)
        if gfi > 0:
            actions.append(f"Look for issues labeled `good first issue` in this repo — {gfi} are currently open")
        actions.append(f"Star the repo and explore the codebase to understand its architecture before contributing")
        actions.append(f"Open a draft PR with improvements to documentation or test coverage to introduce yourself")
        if stars > 1000:
            actions.append(f"Review open issues and comment with your analysis to demonstrate expertise in {domain}")
        else:
            actions.append(f"This smaller repo ({stars:,} stars) is ideal — your contributions will have high impact")
    elif source == "reddit":
        actions.append(f"Add a detailed comment sharing your personal experience with {domain} — be specific, not generic")
        actions.append(f"If you've solved a related problem, link to a GitHub repo or write-up as supporting evidence")
        actions.append(f"Engage with top commenters by asking follow-up questions to build community presence")
    else:
        actions.append(f"Write a substantive comment with your technical perspective — HN rewards depth over breadth")
        actions.append(f"If you have a related project, add it as a Show HN reply with benchmarks or demo link")
        actions.append(f"Track this thread's discussion and return to add follow-up context as the conversation evolves")

    return actions
