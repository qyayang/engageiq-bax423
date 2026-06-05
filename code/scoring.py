"""
Composite engagement scoring.
Implements the 5-component score used in the multi-stage ranking pipeline.
"""
import math
import numpy as np

# Default weights
W_RELEVANCE  = 0.35
W_COMMUNITY  = 0.20
W_VISIBILITY = 0.15
W_EFFORT_INV = 0.10
W_TREND      = 0.20

# Intent mode derived from role — drives soft boosts without hardcoding names
ROLE_INTENT = {
    "ML Student":      "contribution",
    "Data Engineer":   "contribution",
    "DevOps Engineer": "community_engagement",
    "Data Journalist": "trend_spotting",
    "Startup Founder": "startup_growth",
    "Other":           None,
}

# Per-persona weight overrides
PERSONA_WEIGHTS = {
    "ML Student": {
        "relevance": 0.38, "community": 0.18, "visibility": 0.15,
        "effort_inv": 0.14, "trend": 0.15,
    },
    "DevOps Engineer": {
        "relevance": 0.33, "community": 0.35, "visibility": 0.14,
        "effort_inv": 0.05, "trend": 0.13,
    },
    "Data Journalist": {
        "relevance": 0.18, "community": 0.12, "visibility": 0.15,
        "effort_inv": 0.05, "trend": 0.50,
    },
    "Startup Founder": {
        "relevance": 0.38, "community": 0.22, "visibility": 0.15,
        "effort_inv": 0.10, "trend": 0.15,
    },
    "Data Engineer": {
        "relevance": 0.40, "community": 0.25, "visibility": 0.14,
        "effort_inv": 0.11, "trend": 0.10,
    },
}


def _safe_log_norm(x: float, scale: float = 1.0) -> float:
    return min(math.log1p(max(x, 0)) / math.log1p(scale), 1.0)


def compute_community_health(row: dict) -> float:
    source = row.get("source", "") or ""
    if source == "github":
        contrib = _safe_log_norm(row.get("contributors", 0), 200)
        issues  = _safe_log_norm(row.get("open_issues", 0), 500)
        growth  = _safe_log_norm(row.get("growth_rate", 0), 200)
        return round(0.40 * contrib + 0.30 * issues + 0.30 * growth, 4)
    elif source == "reddit":
        comments = _safe_log_norm(row.get("comments", 0), 300)
        upvotes  = _safe_log_norm(row.get("upvotes", 0), 5000)
        return round(0.50 * comments + 0.50 * upvotes, 4)
    elif source == "hackernews":
        comments = _safe_log_norm(row.get("comments", 0), 200)
        upvotes  = _safe_log_norm(row.get("upvotes", 0), 1000)
        return round(0.40 * comments + 0.60 * upvotes, 4)
    return 0.0


def compute_visibility_potential(row: dict) -> float:
    source = row.get("source", "") or ""
    if source == "github":
        stars_norm = _safe_log_norm(row.get("stars", 0), 50000)
        contrib    = max(row.get("contributors", 1), 1)
        stars      = max(row.get("stars", 0), 0)
        gap        = min(stars / contrib / 1000, 1.0)
        return round(0.60 * stars_norm + 0.40 * gap, 4)
    else:
        return round(_safe_log_norm(row.get("upvotes", 0) + row.get("comments", 0), 5000), 4)


def compute_effort_score(row: dict) -> float:
    source = row.get("source", "") or ""
    if source == "github":
        gfi = row.get("good_first_issues", 0)
        if gfi > 0:
            return 0.2
        stars = row.get("stars", 0)
        if stars < 500:   return 0.3
        elif stars < 5000: return 0.5
        else:              return 0.7
    else:
        words = len(str(row.get("description", "")).split())
        return min(words / 300, 0.8)


def compute_trend_score(row: dict) -> float:
    """Recency + velocity signal. Weighted for HN/Reddit sources."""
    growth   = _safe_log_norm(row.get("growth_rate", 0) or 0, 200)
    upvotes  = _safe_log_norm(row.get("upvotes", 0) or 0, 5000)
    comments = _safe_log_norm(row.get("comments", 0) or 0, 500)
    if (row.get("source") or "") in ("hackernews", "reddit"):
        return round(0.35 * growth + 0.38 * upvotes + 0.27 * comments, 4)
    else:
        return round(0.70 * growth + 0.20 * upvotes + 0.10 * comments, 4)


def _intent_bonus(row: dict, intent: str | None) -> float:
    """
    Soft per-intent boost/penalty. Works on any role that maps to an intent,
    including unknown roles — no persona name is hardcoded here.
    """
    if intent is None:
        return 0.0

    source = row.get("source", "") or ""
    lang   = (row.get("language") or "").lower()
    domain = row.get("domain", "") or ""
    bonus  = 0.0

    if intent == "contribution":
        # Boost: GFI, GitHub issues, Python/Jupyter
        if row.get("good_first_issues", 0) > 0:
            bonus += 0.18
        if row.get("record_type", "") == "issue":
            bonus += 0.08
        if lang in ("python", "jupyter notebook", "r", "julia"):
            bonus += 0.05
        # Soft penalty: C/C++/Rust (harder entry barrier)
        if lang in ("c", "c++", "cpp", "rust"):
            bonus -= 0.12

    elif intent == "trend_spotting":
        # Boost: velocity, HN/Reddit discussion
        growth = row.get("growth_rate", 0)
        bonus += min(growth / 130, 0.20)   # up to +0.20 for high-growth
        if source in ("hackernews", "reddit"):
            bonus += 0.10
        bonus += min(row.get("upvotes", 0) / 5000, 0.06)

    elif intent == "community_engagement":
        # Boost: high community health, low contributor saturation
        community = compute_community_health(row)
        bonus += 0.08 * community
        if source == "github":
            contrib = max(row.get("contributors", 1), 1)
            stars   = max(row.get("stars", 0), 0)
            gap     = min(stars / contrib / 1000, 1.0)
            bonus  += 0.06 * gap  # niche/underserved projects

    elif intent == "startup_growth":
        # Boost: API/CLI/tool keywords, SaaS domains, Reddit/HN discussion
        if source in ("reddit", "hackernews"):
            bonus += 0.08
        text = (row.get("title", "") + " " + row.get("description", "")).lower()
        if any(kw in text for kw in ("api", "cli", "sdk", "saas", "productivity", "tool", "platform")):
            bonus += 0.06
        if domain in ("Developer Tools", "B2B SaaS", "Cloud APIs"):
            bonus += 0.05

    return max(-0.15, min(0.22, bonus))


def compute_composite_score(row: dict, relevance: float, persona: str = "") -> dict:
    """
    Returns full score breakdown for one opportunity.
    Formula:
      composite = w_rel·relevance + w_com·community + w_vis·visibility
                + w_eff·effort_inv + w_trend·trend + intent_bonus
    """
    community   = compute_community_health(row)
    visibility  = compute_visibility_potential(row)
    effort      = compute_effort_score(row)
    trend       = compute_trend_score(row)
    effort_inv  = 1.0 - effort

    pw     = PERSONA_WEIGHTS.get(persona, {})
    w_rel  = pw.get("relevance",  W_RELEVANCE)
    w_com  = pw.get("community",  W_COMMUNITY)
    w_vis  = pw.get("visibility", W_VISIBILITY)
    w_eff  = pw.get("effort_inv", W_EFFORT_INV)
    w_trend= pw.get("trend",      W_TREND)

    intent    = ROLE_INTENT.get(persona, None)
    composite = (w_rel * relevance + w_com * community + w_vis * visibility
                 + w_eff * effort_inv + w_trend * trend
                 + _intent_bonus(row, intent))
    composite = max(0.0, min(1.0, composite))

    return {
        "composite_score":  round(composite, 4),
        "relevance_score":  round(relevance, 4),
        "community_health": round(community, 4),
        "visibility_score": round(visibility, 4),
        "effort_score":     round(effort, 4),
        "trend_score":      round(trend, 4),
    }


def build_score_explanation(row: dict, scores: dict) -> str:
    parts = []
    rel  = scores["relevance_score"]
    comm = scores["community_health"]
    vis  = scores["visibility_score"]
    eff  = scores["effort_score"]
    trend= scores.get("trend_score", 0)

    parts.append(f"**Relevance to your interests:** {rel:.0%}")

    source = row.get("source", "")
    if source == "github":
        gfi    = row.get("good_first_issues", 0)
        contrib= row.get("contributors", 0)
        stars  = row.get("stars", 0)
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
        comments= row.get("comments", 0)
        upvotes = row.get("upvotes", 0)
        parts.append(f"**Thread activity:** {comments} comments, {upvotes} upvotes — {comm:.0%} community health")
        parts.append(f"**Visibility:** Your comment reaches {vis:.0%} of potential audience")
        parts.append(f"**Trend score:** {trend:.0%} — velocity signal")
    else:
        comments= row.get("comments", 0)
        score   = row.get("upvotes", 0)
        parts.append(f"**HN score:** {score} points, {comments} comments — {comm:.0%} engagement health")
        parts.append(f"**Trend score:** {trend:.0%} — story velocity")

    parts.append(f"\n**Overall engagement score: {scores['composite_score']:.0%}**")
    return "\n\n".join(parts)


def suggest_actions(row: dict) -> list[str]:
    source = row.get("source", "")
    domain = row.get("domain", "")
    actions = []

    if source == "github":
        gfi   = row.get("good_first_issues", 0)
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
