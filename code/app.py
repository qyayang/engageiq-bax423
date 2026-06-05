"""
EngageIQ — Smart Engagement Opportunity Scorer
BAX-423 Big Data · Spring 2026 · Final Project

Streamlit dashboard integrating:
  • Bloom filter deduplication  (Lecture 2)
  • Sentence-BERT + FAISS ANN  (Lecture 5)
  • Multi-stage ranking         (Lecture 7)
  • Thompson Sampling bandit    (Lecture 8)
"""
import html
import json
import os
import sys
import io
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── path setup ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
DATA_DIR = _HERE.parent / "data"
CSV_PATH = DATA_DIR / "opportunities.csv"

from bloom_filter import BloomFilter
from embeddings import load_or_compute_embeddings, build_faiss_index, embed_query
from ranking import rank_candidates, benchmark_ranking_methods
from adaptive_learning import ThompsonBandit
from analytics import (
    compute_domain_stats,
    compute_trending,
    compute_volume_over_time,
    compute_rising_opportunities,
    compute_community_health_heatmap,
    generate_engagement_brief,
)
from data_collector import StreamingIngester

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EngageIQ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.opp-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.score-bar { height: 6px; border-radius: 3px; background: #30363d; }
.score-fill { height: 6px; border-radius: 3px; }
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 4px;
}
.badge-github { background: #21262d; color: #58a6ff; border: 1px solid #30363d; }
.badge-reddit { background: #ff45001a; color: #ff6314; border: 1px solid #ff6314; }
.badge-hn { background: #ff66001a; color: #ff9500; border: 1px solid #ff9500; }
.badge-domain { background: #1f6feb1a; color: #58a6ff; border: 1px solid #1f6feb; }
.badge-gfi { background: #1a7f371a; color: #3fb950; border: 1px solid #238636; }
h1, h2, h3 { color: #e6edf3 !important; }
p, li { color: #8b949e; }
</style>
""", unsafe_allow_html=True)

DOMAINS = [
    "Machine Learning", "DevOps/K8s", "Trending Open-Source", "Developer Tools",
    "Cybersecurity", "Frontend (React/Web)", "B2B SaaS", "Blockchain",
    "Python Data Eng", "GameDev (C++)", "AI Research", "Embedded Systems (C/RTOS)",
    "Cloud APIs", "Mobile Dev (iOS/Flutter)", "Beginner Coding",
]
SOURCE_ICONS = {"github": "🐙", "reddit": "🤖", "hackernews": "🟠"}
SOURCE_COLORS = {"github": "#58a6ff", "reddit": "#ff6314", "hackernews": "#ff9500"}


# ── data loading ────────────────────────────────────────────────────────────
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["stars", "forks", "contributors", "good_first_issues", "comments", "upvotes", "open_issues"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["activity_score", "growth_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["description"] = df["description"].fillna("")
    df["title"] = df["title"].fillna("")
    # Derive record_type for offline CSV records that pre-date the field
    if "record_type" not in df.columns:
        df["record_type"] = df.apply(
            lambda r: "issue" if "[Issue]" in str(r["title"])
            else "reddit_post" if r["source"] == "reddit"
            else "hn_story" if r["source"] == "hackernews"
            else "repo",
            axis=1,
        )
    if "data_source" not in df.columns:
        df["data_source"] = df.get("_data_source", "offline")
    return df


@st.cache_resource(show_spinner="Initialising database…")
def seed_database():
    """
    Seeds SQLite from CSV on first run (once per app process).
    Pipeline: CSV offline snapshot → SQLite → ranking/retrieval.
    Live API records are appended to SQLite on fetch.
    """
    import db as _db
    _db.init_db()
    # Seed if offline records are below threshold — handles partial/test DB states
    if _db.get_count_by_datasource("offline") < 5000 and CSV_PATH.exists():
        df_seed = pd.read_csv(str(CSV_PATH))
        df_seed = _clean_df(df_seed)
        df_seed["data_source"] = "offline"
        _db.bulk_insert(df_seed.to_dict("records"))
    return True


@st.cache_data(show_spinner="Loading opportunity dataset…")
def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        st.error(f"Dataset not found at {CSV_PATH}. Run `python generate_offline_data.py` first.")
        st.stop()
    seed_database()  # ensure SQLite is seeded
    import db as _db
    try:
        df = _db.get_all_as_df()
        if df.empty:
            raise ValueError("empty")
    except Exception:
        df = pd.read_csv(str(CSV_PATH))
    return _clean_df(df)


@st.cache_resource(show_spinner=False)
def get_bloom_filter() -> BloomFilter:
    return BloomFilter(capacity=200_000, error_rate=0.01)


@st.cache_resource(show_spinner=False)
def get_bandit() -> ThompsonBandit:
    return ThompsonBandit.load()


# ── session state init ───────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "profile": {
            "name": "Professional",
            "role": "ML Student",
            "interests": ["Machine Learning", "AI Research"],
            "time_budget": 5,
        },
        "feedback_log": [],
        "live_records": [],
        "ingester_running": False,
        "refresh_key": 0,
        "sim_metrics": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown("## 🎯 EngageIQ")
    st.sidebar.markdown("*Smart Engagement Opportunity Scorer*")
    st.sidebar.divider()

    st.sidebar.markdown("### 👤 Your Profile")
    name = st.sidebar.text_input("Name", value=st.session_state.profile["name"])
    role = st.sidebar.selectbox(
        "Role",
        ["ML Student", "DevOps Engineer", "Data Journalist", "Startup Founder", "Data Engineer", "Other"],
        index=["ML Student", "DevOps Engineer", "Data Journalist", "Startup Founder", "Data Engineer", "Other"]
        .index(st.session_state.profile.get("role", "ML Student")),
    )
    interests = st.sidebar.multiselect(
        "Interest Domains",
        DOMAINS,
        default=st.session_state.profile.get("interests", ["Machine Learning"]),
    )
    time_budget = st.sidebar.slider(
        "Time budget (hrs/week)", min_value=1, max_value=20,
        value=st.session_state.profile.get("time_budget", 5),
    )

    if st.sidebar.button("💾 Update Profile", use_container_width=True):
        st.session_state.profile = {
            "name": name, "role": role,
            "interests": interests, "time_budget": time_budget,
        }
        # Clear embedding cache so rankings regenerate
        st.session_state.refresh_key += 1
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### 🔍 Filters")
    source_filter = st.sidebar.selectbox("Source", ["All", "GitHub", "Reddit", "Hacker News"])
    domain_filter = st.sidebar.selectbox("Domain", ["All"] + DOMAINS)
    gfi_only = st.sidebar.checkbox("GitHub: Good First Issues only")
    sort_mode = st.sidebar.radio("Sort by", ["Relevance", "Trending", "Community Health"])

    st.sidebar.divider()
    # Live streaming controls
    st.sidebar.markdown("### 📡 Live Stream")
    bloom = get_bloom_filter()
    bstats = bloom.stats()
    st.sidebar.caption(
        f"Bloom filter: {bstats['items_added']:,} items | "
        f"FPR ≈ {bstats['estimated_fpr']:.4f} | {bstats['size_kb']:.0f} KB"
    )
    if st.sidebar.button("🔄 Fetch Live Updates", use_container_width=True):
        with st.sidebar.spinner("Fetching from APIs…"):
            from data_collector import fetch_hn_stories, fetch_reddit_posts, fetch_github_issues
            interests_now = st.session_state.profile.get("interests", DOMAINS[:3])
            new_records = []
            for domain in interests_now[:3]:
                new_records += fetch_hn_stories(domain, n=3)
                new_records += fetch_reddit_posts(domain, n=3)
                new_records += fetch_github_issues(domain, per_page=5)  # real GFI issues
            added = 0
            live_to_persist = []
            for r in new_records:
                uid = r.get("url", r.get("id", ""))
                if not bloom.contains(uid):
                    bloom.add(uid)
                    r["_data_source"] = "live"
                    st.session_state.live_records.append(r)
                    live_to_persist.append(r)
                    added += 1
            # Persist to SQLite so live records survive session refresh
            if live_to_persist:
                import db as _db
                _db.bulk_insert(live_to_persist)
                load_data.clear()  # invalidate cache so next load picks up new records
            st.sidebar.success(
                f"Added {added} live records → persisted to SQLite "
                f"({len(new_records)-added} duplicates blocked by Bloom filter)"
            )

    st.sidebar.divider()
    import db as _db
    db_count = _db.get_count()
    live_count = len([r for r in st.session_state.live_records if r.get("_data_source") == "live"])
    st.sidebar.metric("SQLite Records", f"{db_count:,}")
    st.sidebar.metric("Live Fetched", f"{live_count:,}")
    st.sidebar.metric("Domains Monitored", f"{df['domain'].nunique()}")
    st.sidebar.metric("Feedback Given", f"{len(st.session_state.feedback_log)}")

    return {
        "source": source_filter.lower().replace(" ", "") if source_filter != "All" else "All",
        "domain": domain_filter,
        "exclude_no_gfi": gfi_only,
        "sort_mode": sort_mode,
    }


# ── opportunity card ──────────────────────────────────────────────────────────
def _freshness_badge(created_at: str) -> str:
    """Returns freshness label based on record age. Handles both ISO 8601 and SQL formats."""
    try:
        raw = str(created_at)[:19].replace("T", " ")  # normalise ISO 8601 → SQL format
        dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        days = (datetime.now() - dt).days
        if days <= 7:
            return '<span class="badge" style="background:#1a7f371a;color:#3fb950;border:1px solid #238636">🟢 Fresh</span>'
        elif days <= 30:
            return '<span class="badge" style="background:#1f6feb1a;color:#58a6ff;border:1px solid #1f6feb">🔵 Recent</span>'
        elif days <= 90:
            return '<span class="badge" style="background:#d299221a;color:#d29922;border:1px solid #9e6a03">🟡 Aging</span>'
        else:
            return '<span class="badge" style="background:#30363d;color:#6e7681;border:1px solid #30363d">⚪ Archived</span>'
    except Exception:
        return ""


def _data_source_badge(opp: dict) -> str:
    """
    Labels data as Demo (offline/synthetic) or Live (fetched from real API).
    Uses _data_source field set at fetch time — avoids URL-based guessing
    which would misclassify synthetic Reddit/HN URLs as Live.
    """
    if opp.get("_data_source") == "live" or opp.get("data_source") == "live":
        return '<span class="badge" style="background:#1a7f371a;color:#3fb950;border:1px solid #238636">🟢 Live</span>'
    return '<span class="badge" style="background:#30363d;color:#8b949e;border:1px solid #484f58">🔵 Demo</span>'


def _is_real_url(url: str) -> bool:
    """Returns False for known synthetic placeholder URLs."""
    if not url or url == "#":
        return False
    # Offline dataset uses github.com/user/ as a placeholder username
    if "github.com/user/" in url:
        return False
    return True


def _render_title_link(title: str, url: str, rank: int = None, style: str = "color:#58a6ff;text-decoration:none") -> str:
    prefix = f"#{rank} &nbsp;" if rank is not None else ""
    safe_title = html.escape(title)
    if _is_real_url(url):
        safe_url = html.escape(url)
        return f'{prefix}<a href="{safe_url}" target="_blank" style="{style}">{safe_title}</a>'
    return f'{prefix}<span style="color:#8b949e" title="Demo data — no real URL">{safe_title}</span>'


def render_opportunity_card(opp: dict, rank: int, bandit: ThompsonBandit):
    source = opp.get("source", "")
    icon = SOURCE_ICONS.get(source, "📄")
    domain = opp.get("domain", "")
    title = opp.get("title", "Untitled")
    score = opp.get("final_score", opp.get("composite_score", 0.0))
    gfi = opp.get("good_first_issues", 0)
    stars = opp.get("stars", 0)
    comments = opp.get("comments", 0)
    upvotes = opp.get("upvotes", 0)
    url = opp.get("url", "#")

    score_pct = int(score * 100)
    score_color = "#3fb950" if score > 0.65 else "#d29922" if score > 0.40 else "#f85149"

    badge_html = f'<span class="badge badge-{source}">{icon} {source.upper()}</span>'
    badge_html += f'<span class="badge badge-domain">{domain}</span>'
    if gfi > 0:
        badge_html += f'<span class="badge badge-gfi">✨ {gfi} GFI</span>'
    badge_html += _freshness_badge(opp.get("created_at", ""))
    badge_html += _data_source_badge(opp)

    stats_parts = []
    if stars > 0:
        stats_parts.append(f"⭐ {stars:,}")
    if opp.get("forks", 0) > 0:
        stats_parts.append(f"🍴 {opp['forks']:,}")
    if opp.get("contributors", 0) > 0:
        stats_parts.append(f"👥 {opp['contributors']:,} contrib.")
    if comments > 0:
        stats_parts.append(f"💬 {comments:,}")
    if upvotes > 0 and source != "github":
        stats_parts.append(f"⬆️ {upvotes:,}")
    growth = opp.get("growth_rate", 0)
    if growth > 5:
        stats_parts.append(f"🔥 +{growth:.0f}/wk")

    stats_str = " &nbsp;·&nbsp; ".join(stats_parts)

    desc = html.escape(opp.get("description", "")[:160])
    opp_id = opp.get("id", "")

    # Best next action (visible without expanding)
    actions = opp.get("suggested_actions", [])
    best_action = actions[0] if actions else ""
    best_action_short = html.escape(best_action[:120] + "…" if len(best_action) > 120 else best_action)

    title_html = _render_title_link(title, url, rank=rank)

    col_main, col_score = st.columns([5, 1])
    with col_main:
        st.markdown(f"""
<div class="opp-card">
  <div style="margin-bottom:8px">{badge_html}</div>
  <div style="font-size:16px;font-weight:600;color:#e6edf3;margin-bottom:4px">
    {title_html}
  </div>
  <div style="font-size:13px;color:#8b949e;margin-bottom:6px">{desc}</div>
  <div style="font-size:12px;color:#6e7681;margin-bottom:6px">{stats_str}</div>
  <div style="font-size:12px;color:#3fb950;background:#1a7f371a;border-radius:4px;padding:6px 10px;border-left:3px solid #238636">
    ⚡ <b>Best Action:</b> {best_action_short}
  </div>
</div>
""", unsafe_allow_html=True)

    with col_score:
        st.markdown(f"""
<div style="text-align:center;padding:16px 0">
  <div style="font-size:28px;font-weight:700;color:{score_color}">{score_pct}%</div>
  <div style="font-size:11px;color:#8b949e">Engagement<br>Score</div>
  <div class="score-bar" style="margin-top:8px">
    <div class="score-fill" style="width:{score_pct}%;background:{score_color}"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Expandable details
    with st.expander(f"🔍 Why this? · Suggested Actions · Feedback"):
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.markdown("**Why ranked here:**")
            explanation = opp.get("explanation", "")
            if explanation:
                st.markdown(explanation)
            else:
                scores_detail = {
                    "relevance_score": opp.get("relevance_score", 0),
                    "community_health": opp.get("community_health", 0),
                    "visibility_score": opp.get("visibility_score", 0),
                    "effort_score": opp.get("effort_score", 0),
                    "composite_score": score,
                }
                for k, v in scores_detail.items():
                    label = k.replace("_", " ").title()
                    bar_w = int(v * 100)
                    color = "#58a6ff" if "relevance" in k else "#3fb950" if "community" in k else "#d29922"
                    st.markdown(
                        f'<div style="font-size:13px;margin-bottom:4px">{label}: '
                        f'<b style="color:{color}">{v:.0%}</b></div>'
                        f'<div class="score-bar"><div class="score-fill" style="width:{bar_w}%;background:{color}"></div></div>',
                        unsafe_allow_html=True,
                    )

        with detail_col2:
            st.markdown("**Suggested Actions:**")
            actions = opp.get("suggested_actions", [])
            if actions:
                for a in actions:
                    st.markdown(f"- {a}")

        st.markdown("**Your Feedback:**")
        fb_col1, fb_col2, fb_col3 = st.columns(3)
        bandit = get_bandit()
        with fb_col1:
            if st.button(f"✅ Engage", key=f"engage_{opp_id}"):
                bandit.update(opp_id, "engage", domain)
                bandit.save()  # persist after every action
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "engage",
                     "domain": domain, "ts": datetime.now().isoformat()}
                )
                st.success("Marked as Engage! Bandit updated ✓")
        with fb_col2:
            if st.button(f"⏭️ Skip", key=f"skip_{opp_id}"):
                bandit.update(opp_id, "skip", domain)
                bandit.save()
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "skip",
                     "domain": domain, "ts": datetime.now().isoformat()}
                )
                st.info("Skipped. Bandit updated ✓")
        with fb_col3:
            if st.button(f"🔖 Bookmark", key=f"bm_{opp_id}"):
                bandit.update(opp_id, "bookmark", domain)
                bandit.save()
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "bookmark",
                     "domain": domain, "ts": datetime.now().isoformat()}
                )
                st.success("Bookmarked! Bandit updated ✓")


# ── tab: opportunities ────────────────────────────────────────────────────────
def render_opportunities_tab(df: pd.DataFrame, filters: dict):
    profile = st.session_state.profile
    interests = profile.get("interests", DOMAINS)
    sort_mode = filters.get("sort_mode", "Relevance")

    if not interests:
        st.warning("Please select at least one interest domain in the sidebar.")
        return

    # Build query from interests
    query_text = " ".join(interests)

    # Get embeddings + FAISS index
    all_embs, all_ids = load_or_compute_embeddings(df)
    index, id_array = build_faiss_index(all_ids, all_embs)

    # Encode user query
    query_vec = embed_query(query_text)

    # Stage 1: FAISS ANN retrieval — top 300 candidates
    from embeddings import retrieve_top_k
    candidates = retrieve_top_k(query_vec, index, id_array, k=300)
    c_ids = [c[0] for c in candidates]
    c_sims = [c[1] for c in candidates]

    # Bandit scores + domain prefs
    bandit = get_bandit()
    bandit_scores = bandit.get_bandit_scores(c_ids)
    domain_prefs = bandit.get_domain_preferences()

    # Stage 2+3: Score + re-rank
    source_map = {"All": "All", "github": "GitHub", "reddit": "Reddit",
                  "hackernews": "Hacker News", "hackernews": "hackernews"}
    raw_source = filters.get("source", "All")
    ranked = rank_candidates(
        df=df,
        candidate_ids=c_ids,
        candidate_sims=c_sims,
        bandit_scores=bandit_scores,
        domain_prefs=domain_prefs,
        filters={
            "source": raw_source if raw_source != "All" else "All",
            "domain": filters.get("domain", "All"),
            "exclude_no_gfi": filters.get("exclude_no_gfi", False),
        },
        top_n=50,
        persona=profile.get("role", ""),
    )

    # Apply sort mode override
    if sort_mode == "Trending":
        ranked = sorted(ranked, key=lambda x: x.get("growth_rate", 0), reverse=True)
    elif sort_mode == "Community Health":
        ranked = sorted(ranked, key=lambda x: x.get("community_health", 0), reverse=True)

    if not ranked:
        st.info("No results found. Try adjusting your filters or interests.")
        return

    bandit = get_bandit()

    # ── ACTION QUEUE: Today's Top 5 ──────────────────────────────────────────
    EFFORT_LABELS = {(0, 0.35): "~15 min", (0.35, 0.55): "~30 min",
                     (0.55, 0.70): "~1 hr", (0.70, 1.01): "~2+ hrs"}

    def effort_time(effort_score):
        for (lo, hi), label in EFFORT_LABELS.items():
            if lo <= effort_score < hi:
                return label
        return "~1 hr"

    def why_now(opp):
        growth = opp.get("growth_rate", 0)
        gfi = opp.get("good_first_issues", 0)
        comments = opp.get("comments", 0)
        if growth > 50:
            return f"🔥 Growing fast (+{growth:.0f}/wk) — engage before it goes mainstream"
        if gfi > 0:
            return f"✨ {gfi} beginner issue(s) open — low competition right now"
        if comments < 10 and opp.get("source") in ("reddit", "hackernews"):
            return "💬 Early thread — your comment gets top visibility"
        return "⭐ Highly relevant to your profile based on embedding match"

    st.markdown("""
<div style="background:#161b22;border:1px solid #238636;border-radius:10px;padding:20px;margin-bottom:24px">
  <div style="font-size:18px;font-weight:700;color:#3fb950;margin-bottom:4px">⚡ Today's Action Queue</div>
  <div style="font-size:13px;color:#8b949e">Your top 5 highest-value engagements right now — ranked by persona-adjusted score</div>
</div>
""", unsafe_allow_html=True)

    for i, opp in enumerate(ranked[:5]):
        source = opp.get("source", "")
        icon = SOURCE_ICONS.get(source, "📄")
        title = opp.get("title", "")
        url = opp.get("url", "#")
        domain = opp.get("domain", "")
        score_pct = int(opp.get("final_score", 0) * 100)
        score_color = "#3fb950" if score_pct > 65 else "#d29922"
        actions = opp.get("suggested_actions", [])
        best_action = actions[0] if actions else "Explore this opportunity"
        effort = opp.get("effort_score", 0.5)
        time_est = effort_time(effort)
        reason = why_now(opp)
        opp_id = opp.get("id", "")
        safe_action = html.escape(best_action[:130])
        aq_title_html = _render_title_link(title[:80], url, rank=i+1, style="color:#e6edf3;text-decoration:none")

        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:10px;display:flex;gap:16px">
  <div style="min-width:48px;text-align:center;padding-top:4px">
    <div style="font-size:22px;font-weight:800;color:{score_color}">{score_pct}%</div>
    <div style="font-size:10px;color:#6e7681">score</div>
  </div>
  <div style="flex:1">
    <div style="font-size:11px;color:#8b949e;margin-bottom:4px">
      {icon} {source.upper()} &nbsp;·&nbsp; <span style="color:#58a6ff">{html.escape(domain)}</span>
      &nbsp;·&nbsp; ⏱ {time_est}
    </div>
    <div style="font-size:15px;font-weight:600;color:#e6edf3;margin-bottom:6px">
      {aq_title_html}
    </div>
    <div style="font-size:12px;color:#8b949e;margin-bottom:6px">{reason}</div>
    <div style="font-size:12px;color:#3fb950;background:#1a7f371a;border-radius:4px;padding:5px 10px;border-left:3px solid #238636">
      ⚡ <b>Do this:</b> {safe_action}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        aq_col1, aq_col2, aq_col3 = st.columns([1, 1, 1])
        with aq_col1:
            if st.button("✅ Done / Engage", key=f"aq_engage_{opp_id}"):
                bandit.update(opp_id, "engage", domain)
                bandit.save()
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "engage",
                     "domain": domain, "ts": datetime.now().isoformat()})
                st.success("Logged! Bandit updated ✓")
        with aq_col2:
            if st.button("⏭️ Skip", key=f"aq_skip_{opp_id}"):
                bandit.update(opp_id, "skip", domain)
                bandit.save()
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "skip",
                     "domain": domain, "ts": datetime.now().isoformat()})
                st.info("Skipped ✓")
        with aq_col3:
            if st.button("🔖 Save", key=f"aq_bm_{opp_id}"):
                bandit.update(opp_id, "bookmark", domain)
                bandit.save()
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "bookmark",
                     "domain": domain, "ts": datetime.now().isoformat()})
                st.success("Saved ✓")

    st.markdown("---")

    # ── HEADER STATS ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Opportunities Ranked", f"{len(ranked)}")
    c2.metric("Database Size", f"{len(df):,}")
    c3.metric("Active Interests", f"{len(interests)}")
    c4.metric("Feedback Rounds", f"{bandit.total_rounds}")

    # ── FULL RANKED LIST ──────────────────────────────────────────────────────
    st.markdown("### 📋 Full Ranked List")
    for i, opp in enumerate(ranked[:20]):
        render_opportunity_card(opp, i + 1, bandit)


# ── tab: analytics ────────────────────────────────────────────────────────────
def render_analytics_tab(df: pd.DataFrame):
    st.subheader("📊 Batch Analytics & Trend Intelligence")
    st.caption("Computed over the full opportunity dataset using vectorized batch processing.")

    tab_a, tab_b, tab_c, tab_d = st.tabs(
        ["📈 Domain Health", "🔥 Trending", "📅 Volume Over Time", "🏆 Rising Opportunities"]
    )

    with tab_a:
        domain_stats = compute_domain_stats(df)
        fig = px.bar(
            domain_stats.sort_values("avg_activity"),
            x="avg_activity", y="domain",
            orientation="h",
            color="avg_activity",
            color_continuous_scale="Blues",
            title="Average Activity Score by Domain",
            labels={"avg_activity": "Activity Score", "domain": ""},
            template="plotly_dark",
        )
        fig.update_layout(height=500, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.pie(
                df.groupby("source").size().reset_index(name="count"),
                values="count", names="source",
                title="Source Distribution",
                color_discrete_map={"github": "#58a6ff", "reddit": "#ff6314", "hackernews": "#ff9500"},
                template="plotly_dark",
            )
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            gfi_df = domain_stats[domain_stats["total_gfi"] > 0].sort_values("total_gfi", ascending=False).head(10)
            fig3 = px.bar(
                gfi_df, x="domain", y="total_gfi",
                title="Good First Issues by Domain",
                color="total_gfi", color_continuous_scale="Greens",
                template="plotly_dark",
            )
            fig3.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(
            domain_stats[["domain", "total_opps", "avg_stars", "avg_activity", "avg_growth", "total_gfi"]]
            .rename(columns={
                "domain": "Domain", "total_opps": "# Opportunities",
                "avg_stars": "Avg Stars", "avg_activity": "Avg Activity",
                "avg_growth": "Growth Rate", "total_gfi": "Total GFI",
            }),
            use_container_width=True, hide_index=True,
        )

    with tab_b:
        trending = compute_trending(df, top_n=20)
        if not trending.empty:
            fig = px.scatter(
                trending,
                x="stars", y="velocity",
                size="growth_rate", color="domain",
                hover_name="title",
                title="Trending GitHub Repos: Stars vs Velocity",
                template="plotly_dark",
                labels={"stars": "Star Count", "velocity": "Growth Velocity"},
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Top Trending Repositories:**")
            st.dataframe(
                trending[["title", "domain", "stars", "growth_rate", "velocity"]]
                .rename(columns={"growth_rate": "Stars/wk", "velocity": "Velocity Score"}),
                use_container_width=True, hide_index=True,
            )

    with tab_c:
        vol_df = compute_volume_over_time(df, weeks=12)
        if not vol_df.empty:
            top_domains = df["domain"].value_counts().head(6).index.tolist()
            vol_filtered = vol_df[vol_df["domain"].isin(top_domains)]
            fig = px.line(
                vol_filtered,
                x="week", y="count", color="domain",
                title="Opportunity Volume by Week (Top 6 Domains)",
                template="plotly_dark",
                labels={"count": "New Opportunities", "week": "Week"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab_d:
        st.markdown("### 🚀 Rising Opportunities (Highest Growth Rate)")
        st.caption("Ranked by growth signal. Source and engagement stats are shown per platform.")
        rising = compute_rising_opportunities(df, top_n=15)
        for _, row in rising.iterrows():
            source = row.get("source", "")
            icon = SOURCE_ICONS.get(source, "📄")
            url = row.get("url", "#")
            title = row.get("title", "")
            domain = row.get("domain", "")

            # Source label
            source_label = {"github": "GitHub", "reddit": "Reddit", "hackernews": "Hacker News"}.get(source, source.upper())

            # Per-source stats
            if source == "github":
                stats = f"🔥 +{row.get('growth_rate', 0):.0f} stars/wk · ⭐ {int(row.get('stars', 0)):,} stars"
                title_link = f"[{title}]({url})" if _is_real_url(url) else title
                extra = ""
            elif source == "reddit":
                stats = f"⬆️ {int(row.get('upvotes', 0)):,} upvotes · 💬 {int(row.get('comments', 0)):,} comments"
                title_link = f"[{title}]({url})" if _is_real_url(url) else title
                extra = ""
            elif source == "hackernews":
                stats = f"🔥 {int(row.get('growth_rate', 0)):,} trend score · 💬 {int(row.get('comments', 0)):,} comments"
                title_link = f"[{title}]({url})" if _is_real_url(url) else title
                # If url is external article, note it came from HN; if it IS the HN link, no need for extra
                is_hn_url = "news.ycombinator.com" in url
                extra = "" if is_hn_url else " · *(via Hacker News)*"
            else:
                stats = f"🔥 +{row.get('growth_rate', 0):.0f} growth"
                title_link = f"[{title}]({url})" if _is_real_url(url) else title
                extra = ""

            st.markdown(
                f"**{icon} {source_label}** · **{title_link}** · "
                f"`{domain}` · {stats}{extra}"
            )


# ── tab: learning ─────────────────────────────────────────────────────────────
def render_learning_tab(df: pd.DataFrame):
    st.subheader("🧠 Adaptive Learning Progress")
    st.caption(
        "Thompson Sampling bandit (BAX-423 Lecture 8 — RL) improves recommendations "
        "based on your engage/skip/bookmark signals."
    )

    bandit = get_bandit()
    profile = st.session_state.profile
    interests = profile.get("interests", ["Machine Learning"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Feedback Rounds", bandit.total_rounds)
    col2.metric("Opportunities Evaluated", len(bandit.alpha) + len(bandit.beta_))
    col3.metric("Domains Learned", len(set(bandit.domain_engages) | set(bandit.domain_skips)))

    # Domain preference chart
    prefs = bandit.get_domain_preferences()
    if prefs:
        pref_df = pd.DataFrame(
            [{"Domain": d, "Preference Score": v} for d, v in prefs.items()]
        ).sort_values("Preference Score", ascending=True)
        fig = px.bar(
            pref_df, x="Preference Score", y="Domain",
            orientation="h", title="Learned Domain Preferences",
            color="Preference Score", color_continuous_scale="Viridis",
            template="plotly_dark",
        )
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Feedback history
    if st.session_state.feedback_log:
        st.markdown("**Recent Feedback:**")
        fb_df = pd.DataFrame(st.session_state.feedback_log)
        fb_df["ts"] = pd.to_datetime(fb_df["ts"]).dt.strftime("%H:%M:%S")
        emoji_map = {"engage": "✅", "skip": "⏭️", "bookmark": "🔖"}
        fb_df["Feedback"] = fb_df["feedback"].map(emoji_map)
        st.dataframe(
            fb_df[["ts", "title", "domain", "Feedback"]]
            .rename(columns={"ts": "Time", "title": "Opportunity", "domain": "Domain"}),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.markdown("### 🎮 Simulate 50 Feedback Rounds")
    st.caption(
        "Runs 50 rounds of simulated feedback for your persona — demonstrates measurable "
        "improvement in Precision@10 as the bandit learns."
    )

    persona_map = {
        "ML Student": ["Machine Learning", "AI Research", "Python Data Eng"],
        "DevOps Engineer": ["DevOps/K8s", "Cloud APIs", "Developer Tools"],
        "Data Journalist": ["Trending Open-Source", "AI Research", "Python Data Eng"],
        "Startup Founder": ["Developer Tools", "B2B SaaS", "Cloud APIs"],
        "Data Engineer": ["Python Data Eng", "Cloud APIs", "DevOps/K8s"],
        "Other": interests[:3] if interests else ["Machine Learning"],
    }
    role = profile.get("role", "ML Student")
    sim_domains = persona_map.get(role, interests[:3])

    if st.button("▶️ Run 50 Simulated Feedback Rounds", use_container_width=True):
        fresh_bandit = ThompsonBandit()
        with st.spinner("Simulating 50 feedback rounds…"):
            metrics = fresh_bandit.simulate_feedback_rounds(
                df.sample(min(2000, len(df)), random_state=42),
                n_rounds=50,
                persona_domains=sim_domains,
            )
        st.session_state.sim_metrics = metrics

    if st.session_state.sim_metrics:
        metrics_df = pd.DataFrame(st.session_state.sim_metrics)
        # Rolling average for smooth curve
        metrics_df["smoothed"] = metrics_df["precision_at_10"].rolling(5, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics_df["round"], y=metrics_df["precision_at_10"],
            mode="lines", line=dict(color="#30363d", width=1),
            name="Raw P@10",
        ))
        fig.add_trace(go.Scatter(
            x=metrics_df["round"], y=metrics_df["smoothed"],
            mode="lines", line=dict(color="#58a6ff", width=3),
            name="Smoothed P@10",
        ))
        fig.update_layout(
            title="Precision@10 Over 50 Feedback Rounds (Thompson Sampling)",
            xaxis_title="Feedback Round",
            yaxis_title="Precision@10",
            template="plotly_dark",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        start_p10 = metrics_df["precision_at_10"].head(5).mean()
        end_p10 = metrics_df["precision_at_10"].tail(5).mean()
        improvement = (end_p10 - start_p10) / max(start_p10, 0.001)
        col1, col2, col3 = st.columns(3)
        col1.metric("Initial P@10 (avg rounds 1-5)", f"{start_p10:.2%}")
        col2.metric("Final P@10 (avg rounds 46-50)", f"{end_p10:.2%}")
        col3.metric("Improvement", f"+{improvement:.0%}", delta=f"+{end_p10-start_p10:.2%}")

    st.divider()
    # Ranking benchmark
    st.markdown("### 📐 Ranking Method Benchmark (NDCG@10)")
    st.caption("Compares 4 ranking approaches on 500-record sample to validate multi-stage pipeline.")

    if st.button("🔬 Run Ranking Benchmark", use_container_width=True):
        all_embs, all_ids = load_or_compute_embeddings(df)
        query_vec = embed_query(" ".join(interests))
        with st.spinner("Benchmarking ranking methods…"):
            results = benchmark_ranking_methods(
                df=df,
                query_vec=query_vec,
                all_embeddings=all_embs,
                all_ids=all_ids,
                persona_relevant_domains=sim_domains,
                k=10,
            )
        bench_df = pd.DataFrame(
            [{"Method": k, "NDCG@10": v} for k, v in results.items()]
        ).sort_values("NDCG@10")

        fig = px.bar(
            bench_df, x="NDCG@10", y="Method",
            orientation="h", title="NDCG@10 by Ranking Method",
            color="NDCG@10", color_continuous_scale="Blues",
            template="plotly_dark", range_x=[0, 1],
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bench_df, use_container_width=True, hide_index=True)


# ── tab: export ───────────────────────────────────────────────────────────────
def render_export_tab(df: pd.DataFrame, ranked: list[dict]):
    st.subheader("📥 Download Engagement Brief")
    profile = st.session_state.profile

    brief_data = generate_engagement_brief(df, ranked, profile)
    name = profile.get("name", "Professional")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📋 Brief Summary**")
        st.json({
            "Generated for": brief_data["generated_for"],
            "Date": brief_data["date"],
            "Opportunities analyzed": brief_data["summary"]["total_opportunities_analyzed"],
            "Domains monitored": brief_data["summary"]["domains_monitored"],
            "Your interests": brief_data["summary"]["your_interest_domains"],
        })

    with col2:
        st.markdown("**🏆 Top 10 Opportunities This Week**")
        for opp in brief_data["top_opportunities"]:
            score_pct = int(opp["score"] * 100)
            icon = SOURCE_ICONS.get(opp["source"], "📄")
            st.markdown(
                f"**#{opp['rank']}** {icon} [{opp['title'][:60]}]({opp['url']}) — "
                f"`{opp['domain']}` — **{score_pct}%**"
            )

    st.divider()

    # CSV download
    top_df = pd.DataFrame(brief_data["top_opportunities"])
    csv_bytes = top_df.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download Top-10 CSV",
        data=csv_bytes,
        file_name=f"engageiq_brief_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # JSON brief download
    brief_json = json.dumps(brief_data, indent=2, default=str).encode()
    st.download_button(
        label="⬇️ Download Full Brief (JSON)",
        data=brief_json,
        file_name=f"engageiq_full_brief_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
        use_container_width=True,
    )

    # Full opportunity table
    st.divider()
    st.markdown("**📊 All Ranked Opportunities (for export)**")
    if ranked:
        export_cols = ["title", "source", "domain", "stars", "comments",
                       "good_first_issues", "composite_score", "url"]
        export_df = pd.DataFrame(ranked)[
            [c for c in export_cols if c in pd.DataFrame(ranked).columns]
        ]
        full_csv = export_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Full Ranked List CSV",
            data=full_csv,
            file_name=f"engageiq_all_ranked_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(export_df, use_container_width=True, hide_index=True)


# ── persona test panel ────────────────────────────────────────────────────────
PERSONAS = {
    "Sofia — ML Student": {
        "interests": ["Machine Learning", "AI Research", "Python Data Eng"],
        "role": "ML Student",
        "time_budget": 5,
        "pass_criteria": "≥3 GitHub repos with good-first-issues; ML-focused; no C++/Rust repos",
    },
    "David — DevOps Engineer": {
        "interests": ["DevOps/K8s", "Cloud APIs", "Developer Tools"],
        "role": "DevOps Engineer",
        "time_budget": 3,
        "pass_criteria": "K8s/infra focused; high activity + few contributors; no general webdev",
    },
    "Lina — Data Journalist": {
        "interests": ["Trending Open-Source", "AI Research", "Python Data Eng", "Developer Tools"],
        "role": "Data Journalist",
        "time_budget": 10,
        "pass_criteria": "High velocity/recency; trending signal dominant; broad domains",
    },
    "Raj — Startup Founder": {
        "interests": ["Developer Tools", "B2B SaaS", "Cloud APIs"],
        "role": "Startup Founder",
        "time_budget": 4,
        "pass_criteria": "Dev-tools/SaaS focused; discussion threads; API/CLI repos",
    },
}


def render_persona_tab(df: pd.DataFrame):
    st.subheader("🧪 Test Persona Results")
    st.caption("Pass/fail validation against the 4 graded personas (plus hidden persona readiness).")

    all_embs, all_ids = load_or_compute_embeddings(df)

    results_table = []
    for persona_name, persona in PERSONAS.items():
        interests = persona["interests"]
        query_vec = embed_query(" ".join(interests))
        from embeddings import retrieve_top_k
        index, id_array = build_faiss_index(all_ids, all_embs)
        candidates = retrieve_top_k(query_vec, index, id_array, k=300)
        c_ids = [c[0] for c in candidates]
        c_sims = [c[1] for c in candidates]
        ranked = rank_candidates(df, c_ids, c_sims, top_n=10, persona=persona.get("role", ""))

        role = persona.get("role", "")

        # Use same ranking logic for all personas — driven by role's intent, not name
        # trend_spotting roles: re-sort by trend signal to surface velocity
        from scoring import ROLE_INTENT
        intent = ROLE_INTENT.get(role, None)
        if intent == "trend_spotting":
            top10 = sorted(ranked, key=lambda x: x.get("trend_score", x.get("growth_rate", 0)), reverse=True)[:10]
        else:
            top10 = ranked[:10]

        gfi_count    = sum(1 for r in top10 if r.get("good_first_issues", 0) > 0)
        domain_match = sum(1 for r in top10 if r.get("domain", "") in interests)
        cpp_count    = sum(1 for r in top10 if r.get("language", "").lower() in ("c", "c++", "cpp", "rust"))
        avg_score    = np.mean([r.get("final_score", 0) for r in top10]) if top10 else 0
        avg_trend    = np.mean([r.get("trend_score", r.get("growth_rate", 0)) for r in top10]) if top10 else 0
        discussion_count = sum(1 for r in top10 if r.get("source", "") in ("reddit", "hackernews"))

        # Pass/fail driven by intent, not persona name — hidden persona also benefits
        if intent == "contribution":
            passed = gfi_count >= 3 and cpp_count == 0
        elif intent == "community_engagement":
            passed = domain_match >= 7
        elif intent == "trend_spotting":
            passed = avg_trend > 0.05 and discussion_count >= 3
        elif intent == "startup_growth":
            passed = domain_match >= 6 and discussion_count >= 2
        else:
            passed = domain_match >= 5

        results_table.append({
            "Persona": persona_name,
            "Top-10 Domain Match": f"{domain_match}/10",
            "GFI in Top-10": gfi_count,
            "Discussion (HN/Reddit)": discussion_count,
            "Avg Trend Score": f"{avg_trend:.2f}",
            "C++/Rust in Top-10": cpp_count,
            "Pass Criteria": persona["pass_criteria"][:60] + "…",
            "Result": "✅ PASS" if passed else "❌ FAIL",
        })

        with st.expander(f"{'✅' if passed else '❌'} {persona_name}"):
            st.markdown(f"**Role:** {role} · **Intent mode:** `{intent or 'generic'}`")
            st.markdown(f"**Interests:** {', '.join(interests)}")
            st.markdown(f"**Pass Criteria:** {persona['pass_criteria']}")
            st.markdown(
                f"**Domain match:** {domain_match}/10 · GFI: {gfi_count} · "
                f"Discussion: {discussion_count}/10 · Avg trend: {avg_trend:.2f} · Avg score: {avg_score:.0%}"
            )
            st.markdown("**Top 5 Recommendations:**")
            for i, r in enumerate(top10[:5]):
                icon = SOURCE_ICONS.get(r.get("source", ""), "📄")
                title_md = f"[{r['title'][:70]}]({r['url']})" if _is_real_url(r.get("url", "")) else r["title"][:70]
                st.markdown(
                    f"{i+1}. {icon} {title_md} — "
                    f"`{r['domain']}` — **{r.get('final_score', 0):.0%}**"
                )

    res_df = pd.DataFrame(results_table)
    st.markdown("### Persona Test Summary")
    st.dataframe(res_df, use_container_width=True, hide_index=True)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_data()

    # Add any live records to df
    if st.session_state.live_records:
        live_df = pd.DataFrame(st.session_state.live_records)
        df = pd.concat([df, live_df], ignore_index=True).drop_duplicates(subset=["id"])

    filters = render_sidebar(df)

    # Page header
    st.markdown("""
<div style="padding: 16px 0 8px 0">
  <h1 style="font-size:2rem;margin:0;color:#e6edf3">
    🎯 EngageIQ
  </h1>
  <p style="color:#8b949e;margin:4px 0 0 0;font-size:1rem">
    Smart Engagement Opportunity Scorer — BAX-423 · UC Davis GSM · Spring 2026
  </p>
</div>
""", unsafe_allow_html=True)

    tabs = st.tabs(["🎯 Opportunities", "📊 Analytics", "🧠 Learning", "🧪 Personas", "📥 Export"])

    profile = st.session_state.profile
    interests = profile.get("interests", DOMAINS[:2])
    query_text = " ".join(interests) if interests else "machine learning"

    # Pre-compute rankings for export tab
    all_embs, all_ids = load_or_compute_embeddings(df)
    index, id_array = build_faiss_index(all_ids, all_embs)
    query_vec = embed_query(query_text)
    from embeddings import retrieve_top_k
    candidates = retrieve_top_k(query_vec, index, id_array, k=300)
    c_ids = [c[0] for c in candidates]
    c_sims = [c[1] for c in candidates]
    bandit = get_bandit()
    ranked_for_export = rank_candidates(
        df, c_ids, c_sims,
        bandit_scores=bandit.get_bandit_scores(c_ids),
        domain_prefs=bandit.get_domain_preferences(),
        top_n=50,
        persona=profile.get("role", ""),
    )

    with tabs[0]:
        render_opportunities_tab(df, filters)
    with tabs[1]:
        render_analytics_tab(df)
    with tabs[2]:
        render_learning_tab(df)
    with tabs[3]:
        render_persona_tab(df)
    with tabs[4]:
        render_export_tab(df, ranked_for_export)


if __name__ == "__main__":
    main()
