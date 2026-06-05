"""
Batch analytics and trend detection.
Computes aggregate insights over the full opportunity dataset.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def compute_domain_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats per domain: record count, avg stars, avg activity."""
    agg = df.groupby("domain").agg(
        total_opps=("id", "count"),
        avg_stars=("stars", "mean"),
        avg_activity=("activity_score", "mean"),
        avg_growth=("growth_rate", "mean"),
        github_count=("source", lambda x: (x == "github").sum()),
        hn_count=("source", lambda x: (x == "hackernews").sum()),
        total_gfi=("good_first_issues", "sum"),
    ).reset_index()
    agg["avg_stars"] = agg["avg_stars"].round(0).astype(int)
    agg["avg_activity"] = agg["avg_activity"].round(3)
    agg["avg_growth"] = agg["avg_growth"].round(2)
    return agg.sort_values("avg_activity", ascending=False)


def compute_trending(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Trending = high growth_rate in the last 30 days.
    Returns top_n opportunities sorted by velocity (growth_rate / log(age+1)).
    """
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    now = pd.Timestamp.now()
    df["age_days"] = (now - df["created_at"]).dt.days.fillna(365).clip(lower=1)
    df["velocity"] = df["growth_rate"] / np.log1p(df["age_days"])
    return (
        df[df["source"] == "github"]
        .sort_values("velocity", ascending=False)
        .head(top_n)[["title", "domain", "stars", "growth_rate", "velocity", "url"]]
    )


def compute_volume_over_time(df: pd.DataFrame, weeks: int = 12) -> pd.DataFrame:
    """Returns weekly opportunity ingestion counts per domain."""
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["week"] = df["created_at"].dt.to_period("W").astype(str)
    counts = df.groupby(["week", "domain"]).size().reset_index(name="count")
    # Keep last N weeks
    latest_weeks = sorted(counts["week"].unique())[-weeks:]
    return counts[counts["week"].isin(latest_weeks)]


def compute_community_health_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Domain × Source activity_score matrix for heatmap visualization."""
    pivot = df.pivot_table(
        values="activity_score",
        index="domain",
        columns="source",
        aggfunc="mean",
        fill_value=0,
    ).round(3)
    return pivot


def compute_top_opportunities_per_domain(df: pd.DataFrame, n_per_domain: int = 3) -> pd.DataFrame:
    """Best opportunity per domain by activity_score."""
    return (
        df.sort_values("activity_score", ascending=False)
        .groupby("domain")
        .head(n_per_domain)
        [["domain", "source", "title", "stars", "activity_score", "good_first_issues", "url"]]
        .reset_index(drop=True)
    )


def compute_rising_opportunities(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Opportunities with the highest growth rate — 'rising' signal for trend spotters."""
    return (
        df.sort_values("growth_rate", ascending=False)
        .head(top_n)[["title", "domain", "source", "growth_rate", "stars", "comments", "url"]]
        .reset_index(drop=True)
    )


def generate_engagement_brief(
    df: pd.DataFrame, ranked_results: list[dict], profile: dict
) -> dict:
    """Generates the data payload for the downloadable engagement brief."""
    interests = profile.get("interests", [])
    name = profile.get("name", "Professional")

    top_10 = ranked_results[:10]
    domain_stats = compute_domain_stats(df)
    rising = compute_rising_opportunities(df, top_n=5)

    brief = {
        "title": f"EngageIQ Weekly Engagement Brief",
        "generated_for": name,
        "date": datetime.now().strftime("%B %d, %Y"),
        "summary": {
            "total_opportunities_analyzed": len(df),
            "domains_monitored": len(df["domain"].unique()),
            "your_interest_domains": interests,
        },
        "top_opportunities": [
            {
                "rank": i + 1,
                "title": r["title"],
                "source": r["source"],
                "domain": r["domain"],
                "score": r.get("final_score", r.get("composite_score", 0)),
                "url": r["url"],
            }
            for i, r in enumerate(top_10)
        ],
        "rising_this_week": rising.to_dict("records"),
        "domain_health": domain_stats[["domain", "total_opps", "avg_activity", "total_gfi"]]
        .to_dict("records"),
    }
    return brief
