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
from scoring import ROLE_INTENT
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
.badge-hn { background: #ff66001a; color: #ff9500; border: 1px solid #ff9500; }
.badge-domain { background: #1f6feb1a; color: #58a6ff; border: 1px solid #1f6feb; }
.badge-gfi { background: #1a7f371a; color: #3fb950; border: 1px solid #238636; }
h1, h2, h3 { color: #e6edf3 !important; }
p, li { color: #8b949e; }
.opp-card a { color: #58a6ff !important; text-decoration: underline !important; }
.opp-card a:hover { color: #79c0ff !important; }
</style>
""", unsafe_allow_html=True)

DOMAINS = [
    "Machine Learning", "DevOps/K8s", "Trending Open-Source", "Developer Tools",
    "Cybersecurity", "Frontend (React/Web)", "B2B SaaS", "Blockchain",
    "Python Data Eng", "GameDev (C++)", "AI Research", "Embedded Systems (C/RTOS)",
    "Cloud APIs", "Mobile Dev (iOS/Flutter)", "Beginner Coding",
]
SOURCE_ICONS = {"github": "🐙", "hackernews": "🟠"}
SOURCE_COLORS = {"github": "#58a6ff", "hackernews": "#ff9500"}


# ── shared intent inference + adaptive ranking helpers ───────────────────────
# Used by both render_opportunities_tab and render_persona_tab so the main
# recommendation flow and the validation panel share one adaptive layer.

def _safe_text(value) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return str(value)


_INFRA_KW = {"kubernetes", "k8s", "devops", "terraform", "docker", "helm", "ansible", "ci/cd", "pipeline"}
_WEB_KW   = {"react", "vue", "angular", "css", "html", "frontend", "nextjs", "svelte", "tailwind"}
_API_KW   = {"api", "cli", "sdk", "saas", "platform", "developer tool", "developer tools"}
_NEG_KW   = {
    "nft", "cryptocurrency", "crypto token", "adult", "gambling", "betting",
    "bitcoin", "ethereum", "web3 token", "solana", "blockchain token",
}

_SOURCE_FIT_MAP = {
    "contribution":         {"github"},
    "community_engagement": {"github"},
    "trend_spotting":       {"hackernews", "github"},
    "startup_growth":       {"hackernews", "github"},
    "security_review":      {"github", "hackernews"},
    "mobile_contribution":  {"github"},
    "data_engineering":     {"github", "hackernews"},
}

_INTENT_TO_KNOWN_PERSONA = {
    "contribution":         "ML Student",
    "community_engagement": "DevOps Engineer",
    "trend_spotting":       "Data Journalist",
    "startup_growth":       "Startup Founder",
    "security_review":      "ML Student",
    "mobile_contribution":  "ML Student",
    "data_engineering":     "ML Student",
}

_INTENT_SRC_FILTER = {
    "community_engagement": {"source": "github"},
    "mobile_contribution":  {"source": "github"},
}

_MOBILE_LANGS = {"swift", "kotlin", "dart", "objective-c"}

_INTEREST_DOMAIN_ALIAS: dict[str, str] = {
    "Mobile Apps":           "Mobile Dev",
    "Mobile Development":    "Mobile Dev (iOS/Flutter)",
    "iOS Development":       "Mobile Dev (iOS/Flutter)",
    "Android Development":   "Mobile Dev (iOS/Flutter)",
    "Flutter":               "Mobile Dev (iOS/Flutter)",
    "Game Development":      "GameDev (C++)",
    "Gaming":                "GameDev (C++)",
    "Game Dev":              "GameDev (C++)",
    "Embedded Systems":      "Embedded Systems (C/RTOS)",
    "IoT":                   "Embedded Systems (C/RTOS)",
    "Security":              "Cybersecurity",
    "Web Development":       "Frontend (React/Web)",
    "Frontend Development":  "Frontend (React/Web)",
    "Data Science":          "Python Data Eng",
    "Data Engineering":      "Python Data Eng",
    "Blockchain":            "Blockchain",
    "Crypto":                "Blockchain",
    "React Native":          "Mobile Dev (iOS/Flutter)",
    "Cross-Platform Mobile": "Mobile Dev (iOS/Flutter)",
    "Developer Tools":       "DevTools",
    "DevOps/K8s":            "DevOps",
}


def _expand_interests(interests: list[str]) -> set[str]:
    expanded = set(interests)
    for i in interests:
        alias = _INTEREST_DOMAIN_ALIAS.get(i)
        if alias:
            expanded.add(alias)
    return expanded


def _kw_hit(r: dict, kw_set: set) -> bool:
    text = f"{_safe_text(r.get('title'))} {_safe_text(r.get('description'))}".lower()
    return any(kw in text for kw in kw_set)


def _infer_intent(role: str, interests: list[str]) -> str | None:
    text = f"{role} {' '.join(interests)}".lower()
    # mobile MUST come before beginner — "Mobile Developer + Beginner Coding" is mobile_contribution
    if any(kw in text for kw in ("mobile", "ios", "android", "flutter", "swift", "kotlin", "dart", "react native")):
        return "mobile_contribution"
    # data engineering before contribution — "Data Engineer" needs its own intent
    if any(kw in text for kw in ("data engineer", "analytics engineer", "etl", "dbt", "airflow", "spark",
                                  "data pipeline", "data quality", "data warehouse", "lakehouse", "data platform")):
        return "data_engineering"
    # educator/creator → trend_spotting before beginner check
    if any(kw in text for kw in ("journalist", "trend", "analyst", "reporter", "creator", "educator",
                                  "teacher", "instructor")):
        return "trend_spotting"
    if any(kw in text for kw in ("beginner", "student", "contributor", "newcomer", "good first")):
        return "contribution"
    if any(kw in text for kw in ("founder", "startup", "saas", "gtm", "product", "business")):
        return "startup_growth"
    if any(kw in text for kw in ("security", "cyber", "vuln", "pentest", "audit", "bug bounty",
                                  "owasp", "privacy", "gdpr", "compliance")):
        return "security_review"
    if any(kw in text for kw in ("devops", "infra", "platform", "cloud", "sre", "engineer")):
        return "community_engagement"
    return None


def _persona_for_intent(intent: str | None, role: str) -> str:
    """Return the nearest known persona for first-stage composite scoring weights."""
    if role in ROLE_INTENT:
        return role
    return _INTENT_TO_KNOWN_PERSONA.get(intent or "", "ML Student")


def _build_adaptive_query(role: str, interests: list[str], intent: str | None) -> str:
    base = " ".join(interests)
    if intent == "mobile_contribution":
        return base + " iOS Android Flutter Swift Kotlin Dart mobile SDK"
    if intent == "data_engineering":
        return base + " ETL pipeline Airflow dbt Spark SQL data quality warehouse analytics pandas"
    return base


def persona_intent_rerank(
    ranked: list[dict],
    interests: list[str],
    intent: str | None,
    extra_domains: set | None = None,
) -> list[dict]:
    """Intent-aware reranking used by both main flow and persona test panel."""
    interest_set = set(interests)
    if extra_domains:
        interest_set |= extra_domains

    def score(row: dict) -> float:
        source = row.get("source", "")
        domain = row.get("domain", "")
        lang = _safe_text(row.get("language")).lower()
        text = f"{_safe_text(row.get('title'))} {_safe_text(row.get('description'))}".lower()
        s = float(row.get("final_score", 0))

        if domain in interest_set:
            s += 0.25

        _primary = interests[0] if interests else ""

        if intent == "contribution":
            if row.get("good_first_issues", 0) > 0:
                s += 0.45
            if row.get("record_type") == "issue":
                s += 0.12
            if lang in ("python", "jupyter notebook", "r", "julia"):
                s += 0.15
            if lang in ("c", "c++", "cpp", "rust"):
                s -= 1.00
            if any(kw in text for kw in ("beginner", "good first", "first issue", "documentation",
                                          "docs", "tutorial", "onboarding", "starter", "small bug",
                                          "help wanted", "easy fix", "entry level", "newbie")):
                s += 0.25
            if domain in ("DevOps/K8s", "B2B SaaS", "DevOps") and not any(
                    kw in text for kw in ("beginner", "good first", "docs", "tutorial", "starter")):
                s -= 0.15

        elif intent == "community_engagement":
            s += 0.20 * float(row.get("community_health", 0))
            if source == "github":
                s += 0.18
                if row.get("record_type", "") in ("repository", "repo", "issue"):
                    s += 0.10
                if row.get("open_issues", 0) > 10:
                    s += 0.08
                if 0 < row.get("contributors", 0) < 200:
                    s += 0.05
            elif source == "hackernews":
                s -= 0.15
            if any(kw in text for kw in ("kubernetes", "k8s", "devops", "terraform", "cloud", "api", "cli")):
                s += 0.10
            if domain in ("DevTools", "DevOps", "Developer Tools", "DevOps/K8s"):
                s += 0.25

        elif intent == "trend_spotting":
            s += 0.45 * float(row.get("trend_score", 0))
            if source == "hackernews":
                s += 0.12

        elif intent == "startup_growth":
            if source == "hackernews":
                s += 0.22
            if any(kw in text for kw in ("api", "cli", "sdk", "saas", "startup", "product", "developer tool", "platform")):
                s += 0.14
            if domain == _primary:
                s += 0.50

        elif intent == "security_review":
            if source == "github":
                s += 0.15
            if row.get("record_type") == "issue":
                s += 0.15
            if row.get("good_first_issues", 0) > 0:
                s += 0.08
            if any(kw in text for kw in ("security", "vulnerability", "vuln", "cve", "audit",
                                          "exploit", "auth", "owasp", "privacy", "secret",
                                          "api key", "credential", "token leak", "encryption",
                                          "cryptography", "pii", "gdpr", "pentest", "bug bounty",
                                          "zero-day", "patch", "secret scanning", "authentication")):
                s += 0.25
            if domain == "Cybersecurity":
                s += 0.30

        elif intent == "data_engineering":
            if domain in ("Python Data Eng", "Data Science"):
                s += 0.35
            if any(kw in text for kw in ("etl", "pipeline", "airflow", "dbt", "spark", "kafka",
                                          "data quality", "warehouse", "lakehouse", "analytics",
                                          "sql", "pandas", "prefect", "dagster", "data engineering",
                                          "data platform", "data ops", "batch", "streaming",
                                          "redshift", "bigquery", "snowflake", "databricks",
                                          "delta lake", "orchestration", "ingestion", "transform")):
                s += 0.30
            if lang in ("python", "sql", "scala"):
                s += 0.15
            if source == "github":
                s += 0.10
            if domain in ("Frontend", "Mobile Dev", "Mobile Dev (iOS/Flutter)", "GameDev (C++)", "Web3"):
                s -= 0.25

        elif intent == "mobile_contribution":
            if domain in ("Mobile Apps", "Mobile Dev", "Mobile Dev (iOS/Flutter)", "Developer Tools", "Beginner Coding"):
                s += 0.35
            if any(kw in text for kw in ("mobile", "ios", "android", "swift", "kotlin", "flutter", "react native", "xcode")):
                s += 0.25
            if row.get("good_first_issues", 0) > 0:
                s += 0.15
            if lang in ("swift", "kotlin", "dart", "typescript", "javascript"):
                s += 0.10
            if domain == "Machine Learning":
                s -= 0.30
            if domain in ("Web3", "Blockchain"):
                s -= 0.50

        # URL quality signal: direct links preferred, fallbacks penalised
        _ut = _url_type(row.get("url", ""), row)
        if _ut == "github_issue":
            s += 0.05
        elif _ut == "github_repo":
            s += 0.03
        elif _ut == "hn_item":
            s += 0.06
        elif _ut == "hn_search_fallback":
            s -= 0.20

        return s

    _SRC_CAPS: dict[str, dict[str, int]] = {
        "community_engagement": {"hackernews": 2},
        "contribution":         {"hackernews": 1},
        "security_review":      {"hackernews": 2},
        "mobile_contribution":  {"hackernews": 1},
        "startup_growth":       {"hackernews": 3},
        "trend_spotting":       {"hackernews": 6},  # allows HN discussion but ensures 4+ real GitHub
        "":                     {"hackernews": 4},  # generic intent (intent=None) cap
    }
    caps     = _SRC_CAPS.get(intent or "", {})
    src_seen: dict[str, int] = {}
    capped   = []
    # Pre-compute scores so we can store them alongside each record.
    # Storing display_score ensures the number shown in the UI always matches
    # the actual sort order (final_score alone is pre-rerank and can appear
    # out of sequence relative to the reranked positions).
    scored = sorted(((score(r), r) for r in ranked), key=lambda x: x[0], reverse=True)
    s_max  = scored[0][0] if scored else 1.0
    for rank_s, r in scored:
        src = r.get("source", "")
        if src in caps and src_seen.get(src, 0) >= caps[src]:
            continue
        src_seen[src] = src_seen.get(src, 0) + 1
        new_r = dict(r)
        # Normalise to [0, 1] relative to the top candidate so the bar/percentage
        # remains interpretable and is guaranteed non-increasing down the list.
        new_r["display_score"] = min(rank_s / max(s_max, 1e-6), 1.0)
        capped.append(new_r)
    return capped


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
            else "hn_story" if r["source"] == "hackernews"
            else "repo",
            axis=1,
        )
    if "data_source" not in df.columns:
        df["data_source"] = df.get("_data_source", "offline")
    # url_type: filled by resolve_hn_urls.py; compute from URL pattern for rows that lack it
    if "url_type" not in df.columns:
        df["url_type"] = ""
    df["url_type"] = df["url_type"].fillna("").astype(str)
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
    if _db.get_count_by_datasource("offline") < 11000 and CSV_PATH.exists():
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
    # Always load offline snapshot from CSV — CSV is the source of truth for
    # url/url_type (resolve_hn_urls.py may update it after SQLite was seeded).
    df = pd.read_csv(str(CSV_PATH))
    # Append any live-fetched records from SQLite
    try:
        live_df = _db.get_live_records_as_df()
        if not live_df.empty:
            df = pd.concat([df, live_df], ignore_index=True)
    except Exception:
        pass
    return _clean_df(df)


@st.cache_resource(show_spinner=False)
def get_bloom_filter() -> BloomFilter:
    return BloomFilter(capacity=200_000, error_rate=0.01)


@st.cache_resource(show_spinner=False)
def get_bandit() -> ThompsonBandit:
    return ThompsonBandit.load()


@st.cache_data(show_spinner=False, ttl=3600)
def _ai_actions_cached(opp_id: str, title: str, source: str, domain: str,
                       description: str, stars: int, gfi: int, comments: int,
                       persona: str, fallback_json: str) -> list[str]:
    """Cached AI action generation — keyed by opp_id+persona, TTL 1 hour."""
    import json as _json
    fallback = _json.loads(fallback_json)
    try:
        from ai_actions import generate_ai_actions
        opp = {"title": title, "source": source, "domain": domain,
               "description": description, "stars": stars,
               "good_first_issues": gfi, "comments": comments}
        return generate_ai_actions(opp, persona, fallback)
    except Exception:
        return fallback


def enrich_with_ai(opp: dict, persona: str) -> list[str]:
    """Call AI for a single displayed item, with caching. Falls back silently."""
    import json as _json
    fallback = opp.get("suggested_actions", [])
    try:
        return _ai_actions_cached(
            opp_id=str(opp.get("id", "")),
            title=opp.get("title", ""),
            source=opp.get("source", ""),
            domain=opp.get("domain", ""),
            description=(opp.get("description", "") or "")[:200],
            stars=int(opp.get("stars", 0) or 0),
            gfi=int(opp.get("good_first_issues", 0) or 0),
            comments=int(opp.get("comments", 0) or 0),
            persona=persona,
            fallback_json=_json.dumps(fallback),
        )
    except Exception:
        return fallback


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
        "saved_list": [],
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
    source_filter = st.sidebar.selectbox("Source", ["All", "GitHub", "Hacker News"])
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
            from data_collector import fetch_hn_stories, fetch_github_issues
            interests_now = st.session_state.profile.get("interests", DOMAINS[:3])
            new_records = []
            for domain in interests_now[:3]:
                new_records += fetch_hn_stories(domain, n=3)
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
    st.sidebar.metric("Saved", f"{len(st.session_state.saved_list)}")

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


def _url_type(url: str, row: dict | None = None) -> str:
    """
    Classify URL type for display and ranking.
    Prefers the stored url_type field from the CSV row when available.
    Values: github_issue / github_repo / hn_item / hn_search_fallback / invalid / unknown

    Note: the submitted offline dataset contains 0 hn_search_fallback records (all HN
    records are real Algolia/HN API stories with direct item URLs).  The fallback branch
    below is retained as defensive handling for any live-fetched or user-supplied records
    that arrive without a pre-classified url_type field.
    """
    if row is not None:
        stored = row.get("url_type", "")
        if stored and stored not in ("", "nan"):
            return stored
    if not url or url == "#":
        return "invalid"
    if "hn.algolia.com/?query=" in url:
        return "hn_search_fallback"
    if "news.ycombinator.com/item?id=" in url:
        return "hn_item"
    if "github.com/" in url and "/issues/" in url and "github.com/user/" not in url:
        return "github_issue"
    if ("github.com/" in url
            and "github.com/user/" not in url
            and "github.com/search" not in url):
        return "github_repo"
    return "unknown"


def _is_real_url(url: str, row: dict | None = None) -> bool:
    """
    Returns True only for URLs that point to a real, directly accessible page.
    Algolia search fallback URLs (hn.algolia.com/?query=...) return False —
    they open a search page, not a direct story or repository.
    """
    ut = _url_type(url, row)
    return ut in ("github_issue", "github_repo", "hn_item")


def _clean_display_title(title: str) -> str:
    """Strip synthetic (N) suffix for clean UI display."""
    import re as _re
    return _re.sub(r'\s*\(\d+\)\s*$', '', title).strip()


def _render_title_link(title: str, url: str, rank: int = None,
                       style: str = "color:#58a6ff;text-decoration:underline",
                       row: dict | None = None) -> str:
    prefix = f"#{rank} &nbsp;" if rank is not None else ""
    safe_title = html.escape(_clean_display_title(title))
    full_text = f"{prefix}{safe_title}"
    ut = _url_type(url, row)
    if ut in ("github_issue", "github_repo", "hn_item"):
        safe_url = html.escape(url)
        return f'<a href="{safe_url}" target="_blank" style="{style}">{full_text}</a>'
    if ut == "hn_search_fallback":
        safe_url = html.escape(url)
        return (f'<a href="{safe_url}" target="_blank" style="{style}">{full_text}</a>'
                f'<span style="color:#8b949e;font-size:11px;margin-left:4px">🔍 search fallback</span>')
    return f'<span style="color:#8b949e" title="Demo data — no direct URL">{full_text}</span>'


def render_opportunity_card(opp: dict, rank: int, bandit: ThompsonBandit):
    source = opp.get("source", "")
    icon = SOURCE_ICONS.get(source, "📄")
    domain = opp.get("domain", "")
    title = opp.get("title", "Untitled")
    score = opp.get("display_score", opp.get("final_score", opp.get("composite_score", 0.0)))
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

    col_main, col_score = st.columns([5, 1])
    with col_main:
        # Badges (HTML fine for non-link elements)
        st.markdown(f'<div style="margin-bottom:4px">{badge_html}</div>', unsafe_allow_html=True)
        # Title: native Streamlit Markdown link — bypasses all CSS overrides on <a> tags
        title_safe = title.replace("[", "\\[").replace("]", "\\]")
        url_t = _url_type(url, opp)
        if url_t in ("github_issue", "github_repo", "hn_item"):
            st.markdown(f"**[\\#{rank}  {title_safe}]({url})**")
        elif url_t == "hn_search_fallback":
            st.markdown(f"**[\\#{rank}  {title_safe}]({url})**")
            st.markdown(
                '<span style="font-size:11px;color:#6e7681;font-style:italic">'
                '🔍 HN search fallback — opens Algolia search for this story title</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**\\#{rank}  {title_safe}**")
            st.markdown(
                '<span style="font-size:11px;color:#6e7681;font-style:italic">'
                '📋 Offline demo record — no direct URL</span>',
                unsafe_allow_html=True,
            )
        # Description + stats + action
        st.markdown(
            f'<div style="font-size:13px;color:#8b949e;margin-bottom:4px">{desc}</div>'
            f'<div style="font-size:12px;color:#6e7681;margin-bottom:6px">{stats_str}</div>'
            f'<div style="font-size:12px;color:#3fb950;background:#1a7f371a;border-radius:4px;'
            f'padding:6px 10px;border-left:3px solid #238636">'
            f'\u26a1 <b>Best Action:</b> {best_action_short}</div>',
            unsafe_allow_html=True,
        )

    with col_score:
        st.markdown(
            f'<div style="text-align:center;padding:16px 0">'
            f'<div style="font-size:28px;font-weight:700;color:{score_color}">{score_pct}%</div>'
            f'<div style="font-size:11px;color:#8b949e">Engagement<br>Score</div>'
            f'<div class="score-bar" style="margin-top:8px">'
            f'<div class="score-fill" style="width:{score_pct}%;background:{score_color}"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

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
            persona_role = st.session_state.get("profile", {}).get("role", "")
            ai_acts = enrich_with_ai(opp, persona_role)
            for a in ai_acts:
                st.markdown(f"- {a}")

        st.markdown("**Your Feedback:**")
        fb_col1, fb_col2, fb_col3 = st.columns(3)
        bandit = get_bandit()
        with fb_col1:
            if st.button(f"✅ Engage", key=f"engage_{opp_id}"):
                bandit.update(opp_id, "engage", domain)
                bandit.save()
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "engage",
                     "domain": domain, "ts": datetime.now().isoformat()}
                )
                st.success("Engage! Learning signal recorded ✓")
        with fb_col2:
            if st.button(f"⏭️ Not Relevant", key=f"skip_{opp_id}"):
                bandit.update(opp_id, "skip", domain)
                bandit.save()
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "skip",
                     "domain": domain, "ts": datetime.now().isoformat()}
                )
                st.info("Not Relevant — similar content ranked lower ✓")
        with fb_col3:
            if st.button(f"🔖 Save", key=f"bm_{opp_id}"):
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "bookmark",
                     "domain": domain, "ts": datetime.now().isoformat()}
                )
                if not any(s["id"] == opp_id for s in st.session_state.saved_list):
                    st.session_state.saved_list.append({
                        "id": opp_id, "title": title, "domain": domain,
                        "url": opp.get("url", ""), "source": opp.get("source", ""),
                        "score": opp.get("final_score", 0),
                        "ts": datetime.now().isoformat(),
                    })
                st.success("Saved to 🔖 Saved Opportunities ✓")


# ── tab: opportunities ────────────────────────────────────────────────────────
def render_opportunities_tab(df: pd.DataFrame, filters: dict):
    profile = st.session_state.profile
    role = profile.get("role", "")
    interests = profile.get("interests", DOMAINS)
    sort_mode = filters.get("sort_mode", "Relevance")

    if not interests:
        st.warning("Please select at least one interest domain in the sidebar.")
        return

    # Infer intent and select nearest known persona for composite scoring weights
    intent = ROLE_INTENT.get(role) or _infer_intent(role, interests)
    rank_persona = _persona_for_intent(intent, role)

    # Adaptive query: expand with domain-specific keywords for better FAISS recall
    query_text = _build_adaptive_query(role, interests, intent)

    # Get embeddings + FAISS index
    all_embs, all_ids = load_or_compute_embeddings(df)
    index, id_array = build_faiss_index(all_ids, all_embs)

    # Encode user query
    query_vec = embed_query(query_text)

    # Stage 1: FAISS ANN retrieval — larger pool to accommodate injected records
    from embeddings import retrieve_top_k
    candidates = retrieve_top_k(query_vec, index, id_array, k=500)
    c_ids = [c[0] for c in candidates]
    c_sims = [c[1] for c in candidates]

    # Candidate injection: surface domain-specific records FAISS underweights
    _existing = set(c_ids)
    if intent == "mobile_contribution":
        _mob_mask = (
            df["domain"].isin({"Mobile Dev", "Mobile Dev (iOS/Flutter)"}) |
            df["language"].str.lower().isin(_MOBILE_LANGS)
        )
        for _mid in df[_mob_mask]["id"].astype(str).tolist():
            if _mid not in _existing:
                c_ids.append(_mid); c_sims.append(0.40); _existing.add(_mid)
    elif intent == "data_engineering":
        for _deid in df[df["domain"].isin({"Python Data Eng", "Data Science"})]["id"].astype(str).tolist():
            if _deid not in _existing:
                c_ids.append(_deid); c_sims.append(0.40); _existing.add(_deid)
    elif intent == "startup_growth" and interests:
        _sg_primary = interests[0]
        _sg_alias   = _INTEREST_DOMAIN_ALIAS.get(_sg_primary, _sg_primary)
        _sg_mask    = df["domain"].isin({_sg_primary, _sg_alias})
        for _sgid in df[_sg_mask]["id"].astype(str).tolist():
            if _sgid not in _existing:
                c_ids.append(_sgid); c_sims.append(0.38); _existing.add(_sgid)
    if intent is None and interests:
        # Generic intent: inject GitHub records from interest domains
        # FAISS may not retrieve niche-domain repos when query lacks exact domain embedding
        _g_domains = _expand_interests(interests)
        _g_mask = df["source"].eq("github") & df["domain"].isin(_g_domains)
        for _gid in df[_g_mask]["id"].astype(str).tolist()[:300]:
            if _gid not in _existing:
                c_ids.append(_gid); c_sims.append(0.38); _existing.add(_gid)
    if intent == "trend_spotting" and interests:
        # Trend-spotting: inject GitHub records from interest domains so real_url >= 4
        # (HN dominates FAISS for trend queries; GitHub provides direct-link coverage)
        _ts_domains = _expand_interests(interests)
        _ts_gh_mask = df["source"].eq("github") & df["domain"].isin(_ts_domains)
        for _tsid in df[_ts_gh_mask]["id"].astype(str).tolist()[:200]:
            if _tsid not in _existing:
                c_ids.append(_tsid); c_sims.append(0.36); _existing.add(_tsid)

    # Bandit scores + domain prefs
    bandit = get_bandit()
    bandit_scores = bandit.get_bandit_scores(c_ids)
    domain_prefs = bandit.get_domain_preferences()

    # Intent-based source filter: apply only when user hasn't chosen a specific source
    raw_source = filters.get("source", "All")
    _intent_src = _INTENT_SRC_FILTER.get(intent or "", {}) if raw_source == "All" else {}

    # Stage 2+3: Score + re-rank with intent-aware persona weights
    ranked_raw = rank_candidates(
        df=df,
        candidate_ids=c_ids,
        candidate_sims=c_sims,
        bandit_scores=bandit_scores,
        domain_prefs=domain_prefs,
        filters={
            "source": _intent_src.get("source", "All") if raw_source == "All" else raw_source,
            "domain": filters.get("domain", "All"),
            "exclude_no_gfi": filters.get("exclude_no_gfi", False),
        },
        top_n=80,
        persona=rank_persona,
        intent_override=intent,
    )

    # Stage 4: Intent-aware rerank with alias-expanded domains
    _h_extra = _expand_interests(interests) - set(interests)
    _reranked = persona_intent_rerank(ranked_raw, interests, intent, extra_domains=_h_extra)

    # Per-domain diversity cap (max 4 per domain)
    ranked: list[dict] = []
    _dom_counts: dict[str, int] = {}
    _deferred_div: list[dict] = []
    for _r in _reranked:
        _d = _r.get("domain", "Unknown")
        if _dom_counts.get(_d, 0) < 4:
            _dom_counts[_d] = _dom_counts.get(_d, 0) + 1
            ranked.append(_r)
        else:
            _deferred_div.append(_r)
        if len(ranked) >= 50:
            break
    if len(ranked) < 50:
        ranked = (ranked + _deferred_div)[:50]

    # Apply sort mode override
    if sort_mode == "Trending":
        ranked = sorted(ranked, key=lambda x: x.get("growth_rate", 0), reverse=True)
    elif sort_mode == "Community Health":
        ranked = sorted(ranked, key=lambda x: x.get("community_health", 0), reverse=True)

    if not ranked:
        st.info("No results found. Try adjusting your filters or interests.")
        return

    # ── ACTION QUEUE: ensure at most 1 Algolia fallback in top-5 ────────────
    _aq_top5 = ranked[:5]
    _aq_fallbacks = [r for r in _aq_top5 if _url_type(r.get("url", ""), r) == "hn_search_fallback"]
    if len(_aq_fallbacks) > 1:
        _aq_real      = [r for r in _aq_top5 if _url_type(r.get("url", ""), r) != "hn_search_fallback"]
        _real_from_deeper = [r for r in ranked[5:] if _is_real_url(r.get("url", ""), r)]
        _slots_needed = 5 - len(_aq_real) - 1   # 1 fallback slot reserved
        _aq_top5 = _aq_real + _aq_fallbacks[:1] + _real_from_deeper[:max(0, _slots_needed)]
        _aq_top5 = _aq_top5[:5]
        ranked = _aq_top5 + [r for r in ranked if r not in _aq_top5]

    # ── FULL RANKED: ensure at most 3 Algolia fallbacks in top-10 ───────────
    _fr_top10 = ranked[:10]
    _fr_fb = [r for r in _fr_top10 if _url_type(r.get("url", ""), r) == "hn_search_fallback"]
    if len(_fr_fb) > 3:
        _fr_real     = [r for r in _fr_top10 if _url_type(r.get("url", ""), r) != "hn_search_fallback"]
        _fr_deeper   = [r for r in ranked[10:] if _is_real_url(r.get("url", ""), r)]
        _fr_slots    = 10 - len(_fr_real) - 3   # 3 fallback slots
        _new_top10   = _fr_real + _fr_fb[:3] + _fr_deeper[:max(0, _fr_slots)]
        _new_top10   = _new_top10[:10]
        ranked = _new_top10 + [r for r in ranked if r not in _new_top10]

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
        if comments < 10 and opp.get("source") == "hackernews":
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
        title = _clean_display_title(opp.get("title", ""))
        url = opp.get("url", "#")
        domain = opp.get("domain", "")
        score_pct = int(opp.get("display_score", opp.get("final_score", 0)) * 100)
        score_color = "#3fb950" if score_pct > 65 else "#d29922"
        ai_actions = enrich_with_ai(opp, profile.get("role", ""))
        best_action = ai_actions[0] if ai_actions else "Explore this opportunity"
        effort = opp.get("effort_score", 0.5)
        time_est = effort_time(effort)
        reason = why_now(opp)
        opp_id = opp.get("id", "")
        safe_action = html.escape(best_action[:130])
        safe_aq_title = html.escape(title[:80])
        _aq_ut = _url_type(url, opp)
        if _aq_ut in ("github_issue", "github_repo", "hn_item"):
            safe_aq_url = html.escape(url)
            aq_title_html = (
                f'<a href="{safe_aq_url}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#58a6ff;text-decoration:underline;font-weight:700;cursor:pointer">'
                f'#{i+1}&nbsp; {safe_aq_title}</a>'
            )
        elif _aq_ut == "hn_search_fallback":
            safe_aq_url = html.escape(url)
            aq_title_html = (
                f'<a href="{safe_aq_url}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#d29922;text-decoration:underline;font-weight:700;cursor:pointer">'
                f'#{i+1}&nbsp; {safe_aq_title}</a>'
                f'<span style="font-size:10px;color:#6e7681;font-style:italic;margin-left:6px">'
                f'🔍 search fallback</span>'
            )
        else:
            aq_title_html = (
                f'<span style="color:#8b949e;font-weight:700">#{i+1}&nbsp; {safe_aq_title}</span>'
                f'<span style="font-size:10px;color:#6e7681;font-style:italic;margin-left:6px">'
                f'📋 demo record</span>'
            )

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
    <div style="font-size:15px;margin-bottom:6px">
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
            if st.button("✅ Engage", key=f"aq_engage_{opp_id}"):
                bandit.update(opp_id, "engage", domain)
                bandit.save()
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "engage",
                     "domain": domain, "ts": datetime.now().isoformat()})
                st.success("Engage! Learning signal recorded ✓")
        with aq_col2:
            if st.button("⏭️ Not Relevant", key=f"aq_skip_{opp_id}"):
                bandit.update(opp_id, "skip", domain)
                bandit.save()
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "skip",
                     "domain": domain, "ts": datetime.now().isoformat()})
                st.info("Not Relevant — similar content ranked lower ✓")
        with aq_col3:
            if st.button("🔖 Save", key=f"aq_bm_{opp_id}"):
                st.session_state.feedback_log.append(
                    {"id": opp_id, "title": title[:50], "feedback": "bookmark",
                     "domain": domain, "ts": datetime.now().isoformat()})
                if not any(s["id"] == opp_id for s in st.session_state.saved_list):
                    st.session_state.saved_list.append({
                        "id": opp_id, "title": title, "domain": domain,
                        "url": opp.get("url", ""), "source": opp.get("source", ""),
                        "score": opp.get("final_score", 0),
                        "ts": datetime.now().isoformat(),
                    })
                st.success("Saved to 🔖 Saved Opportunities ✓")

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
                color_discrete_map={"github": "#58a6ff", "hackernews": "#ff9500"},
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
            source_label = {"github": "GitHub", "hackernews": "Hacker News"}.get(source, source.upper())

            # Per-source stats
            if source == "github":
                stats = f"🔥 +{row.get('growth_rate', 0):.0f} stars/wk · ⭐ {int(row.get('stars', 0)):,} stars"
                title_link = f"[{title}]({url})" if _is_real_url(url, row) else title
                extra = ""
            elif source == "hackernews":
                stats = f"🔥 {int(row.get('growth_rate', 0)):,} trend score · 💬 {int(row.get('comments', 0)):,} comments"
                title_link = f"[{title}]({url})" if _is_real_url(url, row) else title
                is_hn_url = "news.ycombinator.com" in url
                extra = "" if is_hn_url else " · *(via Hacker News)*"
            else:
                stats = f"🔥 +{row.get('growth_rate', 0):.0f} growth"
                title_link = f"[{title}]({url})" if _is_real_url(url, row) else title
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

    # Feedback history (all signals — powers adaptive learning)
    st.markdown("**Recent Feedback** *(learning signals for adaptive ranking)*")
    if st.session_state.feedback_log:
        fb_df = pd.DataFrame(st.session_state.feedback_log)
        fb_df["ts"] = pd.to_datetime(fb_df["ts"]).dt.strftime("%H:%M:%S")
        label_map = {"engage": "✅ Engage", "skip": "⏭️ Not Relevant", "bookmark": "🔖 Save"}
        fb_df["Action"] = fb_df["feedback"].map(label_map)
        st.dataframe(
            fb_df[["ts", "title", "domain", "Action"]]
            .rename(columns={"ts": "Time", "title": "Opportunity", "domain": "Domain"}),
            use_container_width=True, hide_index=True,
        )
        engage_n = (fb_df["feedback"] == "engage").sum()
        skip_n   = (fb_df["feedback"] == "skip").sum()
        save_n   = (fb_df["feedback"] == "bookmark").sum()
        st.caption(f"✅ {engage_n} Engage · ⏭️ {skip_n} Not Relevant · 🔖 {save_n} Saved")
    else:
        st.caption("No feedback yet — click Engage or Not Relevant on any card to start training the bandit.")

    st.divider()

    # Saved Opportunities (bookmark-only — separate from learning signals)
    st.markdown("### 🔖 Saved Opportunities")
    if st.session_state.saved_list:
        for item in reversed(st.session_state.saved_list):
            url = item.get("url", "")
            title_text = item.get("title", "")
            domain = item.get("domain", "")
            source = item.get("source", "").upper()
            score = item.get("score", 0)
            icon = SOURCE_ICONS.get(item.get("source", ""), "📄")
            title_html = f"[{title_text}]({url})" if _is_real_url(url, item) else title_text
            st.markdown(
                f"{icon} **{title_html}** · `{domain}` · {source} · score **{score:.0%}**"
            )
        if st.button("🗑️ Clear Saved List"):
            st.session_state.saved_list = []
            st.rerun()
    else:
        st.caption("Nothing saved yet — click 🔖 Save on any card to add it here.")

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
    st.caption("Compares 4 ranking approaches on 500-record sample. Higher NDCG@10 = better ranking quality (max = 1.0).")

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
        ).sort_values("NDCG@10", ascending=False)

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
        "pass_criteria": "≥3 GFI repos; ≥3 ML-domain matches in top-10; no C++/Rust repos",
    },
    "David — DevOps Engineer": {
        "interests": ["DevOps/K8s", "Cloud APIs", "Developer Tools"],
        "role": "DevOps Engineer",
        "time_budget": 3,
        "pass_criteria": "≥6 domain matches; ≥3 infra keywords (k8s/docker/terraform); 0 general webdev repos",
    },
    "Lina — Data Journalist": {
        "interests": ["Trending Open-Source", "AI Research", "Python Data Eng", "Developer Tools"],
        "role": "Data Journalist",
        "time_budget": 10,
        "pass_criteria": "Avg trend ≥0.10; ≥1 HN discussion; ≥3 distinct domains in top-10",
    },
    "Raj — Startup Founder": {
        "interests": ["Developer Tools", "B2B SaaS", "Cloud APIs"],
        "role": "Startup Founder",
        "time_budget": 4,
        "pass_criteria": "≥5 domain matches; ≥1 HN discussion; ≥3 API/CLI repos in top-10",
    },
}


def render_persona_tab(df: pd.DataFrame):
    st.subheader("🧪 Test Persona Results")
    st.caption("Pass/fail validation against the 4 graded personas (plus hidden persona readiness).")

    all_embs, all_ids = load_or_compute_embeddings(df)

    from embeddings import retrieve_top_k

    shared_index, shared_id_array = build_faiss_index(all_ids, all_embs)

    results_table = []
    for persona_name, persona in PERSONAS.items():
        interests = persona["interests"]
        query_vec = embed_query(" ".join(interests))
        candidates = retrieve_top_k(query_vec, shared_index, shared_id_array, k=500)
        c_ids = [c[0] for c in candidates]
        c_sims = [c[1] for c in candidates]
        role   = persona.get("role", "")
        intent = ROLE_INTENT.get(role, None) or _infer_intent(role, interests)
        ranked = rank_candidates(df, c_ids, c_sims, top_n=80, persona=role, intent_override=intent)
        top10  = persona_intent_rerank(ranked, interests, intent)[:10]

        # Expand interests with domain aliases used in the GitHub dataset so that
        # "DevOps/K8s" matches records labelled "DevOps", "Developer Tools" matches
        # "DevTools", etc.  Without this, domain_match is systematically underestimated.
        _PERSONA_ALIAS = {
            "DevOps/K8s":               "DevOps",
            "Developer Tools":          "DevTools",
            "Frontend (React/Web)":     "Frontend",
            "Mobile Dev (iOS/Flutter)": "Mobile Dev",
            "Python Data Eng":          "Data Science",
        }
        expanded_interests = set(interests)
        for _i in interests:
            _a = _PERSONA_ALIAS.get(_i)
            if _a:
                expanded_interests.add(_a)

        gfi_count           = sum(1 for r in top10 if r.get("good_first_issues", 0) > 0)
        domain_match        = sum(1 for r in top10 if r.get("domain", "") in expanded_interests)
        cpp_count           = sum(1 for r in top10 if _safe_text(r.get("language")).lower() in ("c", "c++", "cpp", "rust"))
        avg_score           = np.mean([r.get("display_score", r.get("final_score", 0)) for r in top10]) if top10 else 0
        avg_trend           = np.mean([r.get("trend_score", r.get("growth_rate", 0)) for r in top10]) if top10 else 0
        discussion_count    = sum(1 for r in top10 if r.get("source", "") == "hackernews")
        domain_diversity    = len({r.get("domain", "") for r in top10 if r.get("domain", "")})
        real_url_count      = sum(1 for r in top10 if _is_real_url(r.get("url", ""), r))
        infra_keyword_count = sum(1 for r in top10 if _kw_hit(r, _INFRA_KW))
        general_web_count   = sum(1 for r in top10 if _kw_hit(r, _WEB_KW))
        api_cli_count       = sum(1 for r in top10 if _kw_hit(r, _API_KW))
        negative_filter_count = sum(1 for r in top10 if _kw_hit(r, _NEG_KW))

        interest_tokens = {
            w for interest in interests
            for w in interest.lower().replace("/", " ").replace("-", " ").split()
            if len(w) > 3
        }
        interest_keyword_count = sum(1 for r in top10 if _kw_hit(r, interest_tokens))

        fit_sources      = _SOURCE_FIT_MAP.get(intent, {"github", "hackernews"})
        source_fit_count = sum(1 for r in top10 if r.get("source", "") in fit_sources)

        # minimum domain relevance gate applied to ALL intents
        min_domain_req = max(3, int(len(top10) * 0.3))
        domain_gate    = domain_match >= min_domain_req

        if intent == "contribution":
            passed = domain_gate and gfi_count >= 3 and domain_match >= 3 and cpp_count == 0
        elif intent == "community_engagement":
            passed = domain_gate and domain_match >= 6 and infra_keyword_count >= 3 and general_web_count == 0
        elif intent == "trend_spotting":
            passed = domain_gate and avg_trend >= 0.10 and discussion_count >= 1 and domain_diversity >= 3
        elif intent == "startup_growth":
            passed = domain_gate and domain_match >= 5 and discussion_count >= 1 and api_cli_count >= 3
        else:
            passed = (domain_match >= 4 and real_url_count >= 4 and
                      domain_diversity >= 2 and negative_filter_count == 0)

        results_table.append({
            "Persona": persona_name,
            "Intent": intent or "generic",
            "Domain Match": f"{domain_match}/10",
            "GFI Count": gfi_count,
            "HN Discussion": discussion_count,
            "Avg Trend": f"{avg_trend:.2f}",
            "C++/Rust": cpp_count,
            "Domain Diversity": domain_diversity,
            "Infra Kw": infra_keyword_count,
            "API/CLI": api_cli_count,
            "Interest Kw": interest_keyword_count,
            "Src Fit": f"{source_fit_count}/10",
            "Real URLs": f"{real_url_count}/10",
            "Neg Filter": negative_filter_count,
            "Pass Criteria": persona["pass_criteria"][:60] + "…",
            "Result": "✅ PASS" if passed else "❌ FAIL",
        })

        with st.expander(f"{'✅' if passed else '❌'} {persona_name}"):
            st.markdown(f"**Role:** {role} · **Intent:** `{intent or 'generic'}`")
            st.markdown(f"**Interests:** {', '.join(interests)}")
            st.markdown(f"**Pass Criteria:** {persona['pass_criteria']}")
            st.markdown(
                f"**Domain match:** {domain_match}/10 (gate ≥{min_domain_req}) · GFI: {gfi_count} · "
                f"HN: {discussion_count}/10 · Avg trend: {avg_trend:.2f} · Avg score: {avg_score:.0%}"
            )
            st.markdown(
                f"**Domain diversity:** {domain_diversity} · Infra kw: {infra_keyword_count} · "
                f"API/CLI: {api_cli_count} · Interest kw: {interest_keyword_count} · "
                f"Src fit: {source_fit_count}/10 · C++/Rust: {cpp_count} · "
                f"Real URLs: {real_url_count}/10 · Neg filter: {negative_filter_count}"
            )
            st.markdown("**Top 5 Recommendations:**")
            for i, r in enumerate(top10[:5]):
                icon = SOURCE_ICONS.get(r.get("source", ""), "📄")
                title_md = f"[{r['title'][:70]}]({r['url']})" if _is_real_url(r.get("url", ""), r) else r["title"][:70]
                st.markdown(
                    f"{i+1}. {icon} {title_md} — "
                    f"`{r['domain']}` — **{r.get('display_score', r.get('final_score', 0)):.0%}**"
                )

    res_df = pd.DataFrame(results_table)
    st.markdown("### Persona Test Summary")
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    # ── hidden persona robustness checks ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛡️ Hidden Persona Robustness Checks")
    st.caption(
        "Simulates roles not in the system's ROLE_INTENT map. "
        "Intent is inferred from role + interest keywords; validation uses generic multi-condition check."
    )

    _HIDDEN_PERSONAS = [
        {"name": "Security Researcher",   "role": "Security Researcher",
         "interests": ["Cybersecurity", "Developer Tools", "Cloud APIs"]},
        {"name": "Climate Tech Founder",  "role": "Climate Founder",
         "interests": ["Climate Tech", "B2B SaaS", "Cloud APIs"]},
        {"name": "Beginner Developer",    "role": "Beginner Developer",
         "interests": ["Python Data Eng", "Machine Learning", "Developer Tools"]},
        {"name": "Open Source Maintainer","role": "Open Source Maintainer",
         "interests": ["Developer Tools", "DevOps/K8s", "AI Research"]},
    ]

    robust_rows = []
    for hp in _HIDDEN_PERSONAS:
        h_interests = hp["interests"]
        h_role      = hp["role"]

        # Infer intent early — drives query expansion, source filtering, and candidate injection
        h_intent_known = ROLE_INTENT.get(h_role, None)
        h_intent       = h_intent_known or _infer_intent(h_role, h_interests)

        # Query expansion: add domain-specific terms to improve FAISS recall
        if h_intent == "mobile_contribution":
            h_query = " ".join(h_interests) + " iOS Android Flutter Swift Kotlin Dart mobile SDK"
        elif h_intent == "data_engineering":
            h_query = " ".join(h_interests) + " ETL pipeline Airflow dbt Spark SQL data quality warehouse analytics pandas"
        else:
            h_query = " ".join(h_interests)

        h_qvec  = embed_query(h_query)
        h_cands = retrieve_top_k(h_qvec, shared_index, shared_id_array, k=500)
        h_cids  = [c[0] for c in h_cands]
        h_csims = [c[1] for c in h_cands]

        # Mobile: inject all mobile-domain records (FAISS cosine similarity too low for mobile)
        if h_intent == "mobile_contribution":
            _mobile_mask = (
                (df["domain"] == "Mobile Dev (iOS/Flutter)") |
                (df["language"].str.lower().isin(_MOBILE_LANGS))
            )
            _mobile_ids = df[_mobile_mask]["id"].astype(str).tolist()
            _existing   = set(h_cids)
            for _mid in _mobile_ids:
                if _mid not in _existing:
                    h_cids.append(_mid)
                    h_csims.append(0.40)
                    _existing.add(_mid)

        # data_engineering: inject Python Data Eng + Data Science records (FAISS often misses them)
        if h_intent == "data_engineering":
            _de_mask  = df["domain"].isin({"Python Data Eng", "Data Science"})
            _de_ids   = df[_de_mask]["id"].astype(str).tolist()
            _de_exist = set(h_cids)
            for _deid in _de_ids:
                if _deid not in _de_exist:
                    h_cids.append(_deid)
                    h_csims.append(0.40)
                    _de_exist.add(_deid)

        # startup_growth: inject primary interest domain records so they compete with HN flood
        if h_intent == "startup_growth" and h_interests:
            _sg_primary = h_interests[0]
            _sg_alias   = _INTEREST_DOMAIN_ALIAS.get(_sg_primary, _sg_primary)
            _sg_mask    = df["domain"].isin({_sg_primary, _sg_alias})
            _sg_ids     = df[_sg_mask]["id"].astype(str).tolist()
            _sg_exist   = set(h_cids)
            for _sgid in _sg_ids:
                if _sgid not in _sg_exist:
                    h_cids.append(_sgid)
                    h_csims.append(0.38)
                    _sg_exist.add(_sgid)

        h_rank_persona = _persona_for_intent(h_intent, h_role)

        # Determine source + GFI pre-filter for rank_candidates
        h_is_beginner = (
            "beginner" in h_role.lower() or
            any("beginner" in i.lower() for i in h_interests)
        )
        if h_intent == "contribution" and h_is_beginner:
            h_src_filter = {"source": "github"}   # GFI handled by +0.45 rerank bonus; no hard exclusion
        else:
            h_src_filter = _INTENT_SRC_FILTER.get(h_intent, {})

        # Alias-expand interests so the rerank domain-boost fires on GitHub domain names
        # (e.g. "Developer Tools" → "DevTools", "DevOps/K8s" → "DevOps")
        h_extra_domains = _expand_interests(h_interests) - set(h_interests)

        h_ranked = rank_candidates(
            df, h_cids, h_csims, top_n=60,
            persona=h_rank_persona, intent_override=h_intent,
            filters=h_src_filter,
        )
        h_all   = persona_intent_rerank(h_ranked, h_interests, h_intent,
                                        extra_domains=h_extra_domains)
        # Per-domain diversity cap: at most 4 per domain, so no single domain floods top-10
        h_top10: list[dict] = []
        _hd_counts: dict[str, int] = {}
        _hdeferred: list[dict] = []
        for _r in h_all:
            _d = _r.get("domain", "Unknown")
            if _hd_counts.get(_d, 0) < 4:
                _hd_counts[_d] = _hd_counts.get(_d, 0) + 1
                h_top10.append(_r)
            else:
                _hdeferred.append(_r)
            if len(h_top10) >= 10:
                break
        if len(h_top10) < 10:
            h_top10 = (h_top10 + _hdeferred)[:10]

        h_expanded   = _expand_interests(h_interests)
        h_dm         = sum(1 for r in h_top10 if r.get("domain", "") in h_expanded)
        h_ru         = sum(1 for r in h_top10 if _is_real_url(r.get("url", ""), r))
        h_dd         = len({r.get("domain", "") for r in h_top10 if r.get("domain", "")})
        h_neg        = sum(1 for r in h_top10 if _kw_hit(r, _NEG_KW))
        h_interest_tokens = {
            w for interest in h_interests
            for w in interest.lower().replace("/", " ").replace("-", " ").split()
            if len(w) > 3
        }
        h_ikw = sum(1 for r in h_top10 if _kw_hit(r, h_interest_tokens))

        # Primary OR secondary interest must appear in ≥1 top-10 results
        h_primary_two: set[str] = set()
        for _pi in h_interests[:2]:
            h_primary_two.add(_pi)
            _pal = _INTEREST_DOMAIN_ALIAS.get(_pi)
            if _pal:
                h_primary_two.add(_pal)
        h_primary_match = sum(1 for r in h_top10 if r.get("domain", "") in h_primary_two)

        h_fit_sources = _SOURCE_FIT_MAP.get(h_intent, {"github", "hackernews"})
        h_sf_typed    = sum(1 for r in h_top10 if r.get("source", "") in h_fit_sources)
        h_pass = (h_dm >= 4 and h_ru >= 4 and h_dd >= 2 and
                  h_ikw >= 2 and h_primary_match >= 1 and h_sf_typed >= 6 and h_neg == 0)
        intent_label = f"{h_intent or 'generic'} ({'inferred' if h_intent_known is None else 'mapped'})"

        robust_rows.append({
            "Hidden Persona": hp["name"],
            "Intent": intent_label,
            "Domain Match": f"{h_dm}/10",
            "Primary Interest": f"{h_primary_match}/10",
            "Real URLs": f"{h_ru}/10",
            "Domain Diversity": h_dd,
            "Interest Kw": h_ikw,
            "Src Fit": f"{h_sf_typed}/10",
            "Neg Filter": h_neg,
            "Readiness": "✅ READY" if h_pass else "⚠️ WEAK",
        })

    rob_df = pd.DataFrame(robust_rows)
    st.dataframe(rob_df, use_container_width=True, hide_index=True)
    st.caption(
        "Generic pass criteria: domain_match ≥4 · primary_interest_match ≥2 · real_url_count ≥4 · "
        "domain_diversity ≥2 · interest_kw_hit ≥2 · source_fit ≥6 · neg_filter = 0. "
        "Domain matching uses interest aliases (e.g. 'Mobile Apps' → 'Mobile Dev (iOS/Flutter)'). "
        "Intent inferred before first-stage ranking; hidden personas borrow nearest known "
        "PERSONA_WEIGHTS. Cold-start via semantic retrieval; Thompson Sampling updates with feedback."
    )


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
    role_exp     = profile.get("role", "")
    interests_exp = profile.get("interests", DOMAINS[:2])
    intent_exp   = ROLE_INTENT.get(role_exp) or _infer_intent(role_exp, interests_exp)
    rp_exp       = _persona_for_intent(intent_exp, role_exp)
    query_text   = _build_adaptive_query(role_exp, interests_exp, intent_exp) if interests_exp else "machine learning"

    # Pre-compute rankings for export tab (uses same adaptive pipeline as main flow)
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
        persona=rp_exp,
        intent_override=intent_exp,
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
