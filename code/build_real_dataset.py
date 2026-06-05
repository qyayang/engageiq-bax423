"""
Real API data collector — replaces synthetic offline snapshot with actual scraped data.
Sources: GitHub REST API, GH Archive, Reddit public JSON, Hacker News Firebase API.
Run: python build_real_dataset.py
"""
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import praw

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_CSV = DATA_DIR / "opportunities.csv"

_GH_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    _GH_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    print("GitHub token detected — higher rate limit active (5000 req/hr)")
else:
    print("No GitHub token — using 60 req/hr. Set GITHUB_TOKEN env var to speed up.")

DOMAIN_QUERIES = {
    "Machine Learning":          "machine learning pytorch tensorflow neural network",
    "DevOps/K8s":                "kubernetes helm terraform devops CI/CD",
    "Trending Open-Source":      "trending open source stars:>500",
    "Developer Tools":           "developer tools CLI productivity developer-experience",
    "Cybersecurity":             "security vulnerability scanner OWASP pentest",
    "Frontend (React/Web)":      "react component typescript nextjs frontend",
    "B2B SaaS":                  "saas multi-tenant billing stripe subscription",
    "Blockchain":                "smart contract ethereum solidity defi web3",
    "Python Data Eng":           "data pipeline spark airflow dbt data engineering",
    "GameDev (C++)":             "game engine opengl vulkan gamedev C++",
    "AI Research":               "large language model alignment RLHF diffusion transformer",
    "Embedded Systems (C/RTOS)": "embedded rtos freertos microcontroller STM32",
    "Cloud APIs":                "aws gcp azure serverless cloud-native terraform",
    "Mobile Dev (iOS/Flutter)":  "flutter ios swift swiftui mobile app",
    "Beginner Coding":           "good-first-issue beginner tutorial learn-to-code",
}

DOMAIN_SUBREDDITS = {
    "Machine Learning":          ["MachineLearning", "learnmachinelearning"],
    "DevOps/K8s":                ["devops", "kubernetes"],
    "Trending Open-Source":      ["opensource", "programming"],
    "Developer Tools":           ["webdev", "programming"],
    "Cybersecurity":             ["netsec", "cybersecurity"],
    "Frontend (React/Web)":      ["reactjs", "webdev"],
    "B2B SaaS":                  ["SaaS", "startups"],
    "Blockchain":                ["ethereum", "defi"],
    "Python Data Eng":           ["dataengineering", "Python"],
    "GameDev (C++)":             ["gamedev", "cpp"],
    "AI Research":               ["MachineLearning", "LocalLLaMA"],
    "Embedded Systems (C/RTOS)": ["embedded", "RTOS"],
    "Cloud APIs":                ["aws", "googlecloud"],
    "Mobile Dev (iOS/Flutter)":  ["FlutterDev", "iOSProgramming"],
    "Beginner Coding":           ["learnprogramming", "learnpython"],
}

DOMAINS = list(DOMAIN_QUERIES.keys())


def _id(source: str, raw: str) -> str:
    return hashlib.md5(f"{source}:{raw}".encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def activity_score(stars, forks, contributors, comments, growth) -> float:
    import math
    s = min(math.log1p(max(stars, 0)) / math.log1p(50000), 1.0) * 0.25
    f = min(math.log1p(max(forks, 0)) / math.log1p(10000), 1.0) * 0.15
    c = min(math.log1p(max(contributors, 0)) / math.log1p(200), 1.0) * 0.20
    co = min(math.log1p(max(comments, 0)) / math.log1p(500), 1.0) * 0.15
    g = min(max(growth, 0) / 100, 1.0) * 0.25
    return round(s + f + c + co + g, 4)


# ── GitHub Repos ──────────────────────────────────────────────────────────────
def fetch_github_repos_real(domain: str, pages: int = 3) -> list[dict]:
    query = DOMAIN_QUERIES[domain]
    records = []
    for page in range(1, pages + 1):
        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                headers=_GH_HEADERS,
                params={"q": query, "sort": "stars", "order": "desc",
                        "per_page": 100, "page": page},
                timeout=12,
            )
            if resp.status_code == 403:
                print(f"  GitHub rate limit hit — waiting 60s")
                time.sleep(60)
                continue
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                break
            for item in items:
                stars = item.get("stargazers_count", 0)
                forks = item.get("forks_count", 0)
                issues = item.get("open_issues_count", 0)
                lang = item.get("language") or ""
                records.append({
                    "id": _id("github_repo", str(item["id"])),
                    "source": "github", "record_type": "repo", "data_source": "live",
                    "title": item.get("full_name", ""),
                    "description": (item.get("description") or "")[:300],
                    "url": item.get("html_url", ""),
                    "domain": domain, "language": lang,
                    "tags": json.dumps(item.get("topics", [])[:5]),
                    "stars": stars, "forks": forks, "contributors": 0,
                    "open_issues": issues, "good_first_issues": 0,
                    "comments": issues, "upvotes": 0,
                    "activity_score": activity_score(stars, forks, 0, issues, 0),
                    "growth_rate": 0.0,
                    "created_at": (item.get("created_at") or "")[:19].replace("T", " "),
                    "updated_at": (item.get("updated_at") or "")[:19].replace("T", " "),
                })
            time.sleep(0.3 if GITHUB_TOKEN else 2.0)
        except Exception as e:
            print(f"  GitHub repo error ({domain} p{page}): {e}")
            break
    return records


# ── GitHub Issues (GFI) ───────────────────────────────────────────────────────
def fetch_github_issues_real(domain: str, pages: int = 2) -> list[dict]:
    query = DOMAIN_QUERIES[domain]
    search_q = f"{query} label:\"good first issue\" state:open"
    records = []
    for page in range(1, pages + 1):
        try:
            resp = requests.get(
                "https://api.github.com/search/issues",
                headers=_GH_HEADERS,
                params={"q": search_q, "sort": "created", "order": "desc",
                        "per_page": 100, "page": page},
                timeout=12,
            )
            if resp.status_code == 403:
                time.sleep(60)
                continue
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                break
            for item in items:
                repo_url = item.get("repository_url", "")
                repo_name = repo_url.split("repos/")[-1] if "repos/" in repo_url else ""
                labels = [lb["name"] for lb in item.get("labels", [])]
                comments = item.get("comments", 0)
                records.append({
                    "id": _id("github_issue", str(item["id"])),
                    "source": "github", "record_type": "issue", "data_source": "live",
                    "title": f"[Issue] {item.get('title', '')}",
                    "description": f"GFI issue in {repo_name}. Labels: {', '.join(labels)}. {(item.get('body') or '')[:200]}",
                    "url": item.get("html_url", ""),
                    "domain": domain, "language": "",
                    "tags": json.dumps(labels[:5]),
                    "stars": 0, "forks": 0, "contributors": 0,
                    "open_issues": 1, "good_first_issues": 1,
                    "comments": comments, "upvotes": 0,
                    "activity_score": activity_score(0, 0, 0, comments, 0),
                    "growth_rate": 0.0,
                    "created_at": (item.get("created_at") or "")[:19].replace("T", " "),
                    "updated_at": (item.get("updated_at") or "")[:19].replace("T", " "),
                })
            time.sleep(0.3 if GITHUB_TOKEN else 2.0)
        except Exception as e:
            print(f"  GitHub issues error ({domain} p{page}): {e}")
            break
    return records


# ── Reddit ────────────────────────────────────────────────────────────────────
def _get_reddit() -> praw.Reddit | None:
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return None
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent="EngageIQ/1.0 BAX423 (by /u/engageiq_bot)",
    )


def fetch_reddit_real(domain: str, n: int = 50) -> list[dict]:
    reddit = _get_reddit()
    if reddit is None:
        print(f"  Reddit skipped — REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")
        return []

    subs = DOMAIN_SUBREDDITS.get(domain, ["programming"])
    records = []
    for sub in subs:
        for sort in ["hot", "top"]:
            try:
                subreddit = reddit.subreddit(sub)
                posts = list(subreddit.hot(limit=50) if sort == "hot" else subreddit.top("week", limit=50))
                for post in posts:
                    if post.stickied or not post.title:
                        continue
                    score = post.score
                    n_comments = post.num_comments
                    created = datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d %H:%M:%S")
                    records.append({
                        "id": _id("reddit", post.id),
                        "source": "reddit", "record_type": "reddit_post", "data_source": "live",
                        "title": post.title,
                        "description": f"r/{sub} [{sort}] — {n_comments} comments, {score} upvotes. {(post.selftext or '')[:200]}",
                        "url": f"https://reddit.com{post.permalink}",
                        "domain": domain, "language": "",
                        "tags": json.dumps([sub]),
                        "stars": 0, "forks": 0, "contributors": 0,
                        "open_issues": 0, "good_first_issues": 0,
                        "comments": n_comments, "upvotes": score,
                        "activity_score": activity_score(0, 0, 0, n_comments, score / 100),
                        "growth_rate": round(score / 100, 2),
                        "created_at": created, "updated_at": _now(),
                    })
                time.sleep(0.5)
            except Exception as e:
                print(f"  Reddit error (r/{sub} {sort}): {e}")
    return records[:n]


# ── Hacker News ───────────────────────────────────────────────────────────────
def fetch_hn_real(domain: str, n: int = 50) -> list[dict]:
    keywords = DOMAIN_QUERIES[domain].lower().split()
    records = []
    try:
        # Fetch from multiple lists for broader coverage
        story_ids = []
        for endpoint in ["topstories", "newstories", "beststories"]:
            ids = requests.get(
                f"https://hacker-news.firebaseio.com/v0/{endpoint}.json",
                timeout=8,
            ).json()
            story_ids.extend(ids[:200])
        story_ids = list(dict.fromkeys(story_ids))  # dedup, preserve order
    except Exception:
        return []

    for sid in story_ids:
        if len(records) >= n:
            break
        try:
            item = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                timeout=5,
            ).json()
            if not item or item.get("type") not in ("story", "ask", "show"):
                continue
            title = item.get("title", "")
            title_lower = title.lower()
            if not any(kw[:5] in title_lower for kw in keywords[:4]):
                continue
            score = item.get("score", 0)
            n_comments = item.get("descendants", 0)
            ts = item.get("time", 0)
            created = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else _now()
            records.append({
                "id": _id("hn", str(sid)),
                "source": "hackernews", "record_type": "hn_story", "data_source": "live",
                "title": title,
                "description": f"HN {item.get('type','story')} — score: {score}, {n_comments} comments. {item.get('url','')}",
                "url": item.get("url") or f"https://news.ycombinator.com/item?id={sid}",
                "domain": domain, "language": "",
                "tags": json.dumps(keywords[:3]),
                "stars": 0, "forks": 0, "contributors": 0,
                "open_issues": 0, "good_first_issues": 0,
                "comments": n_comments, "upvotes": score,
                "activity_score": activity_score(0, 0, 0, n_comments, score / 50),
                "growth_rate": round(score / 10, 2),
                "created_at": created, "updated_at": _now(),
            })
        except Exception:
            continue
    return records


# ── GH Archive (trending repos from real GitHub events) ──────────────────────
def fetch_gharchive(max_records: int = 2000) -> list[dict]:
    """
    Downloads one hour of GH Archive data and extracts WatchEvent (stars)
    to identify trending repos with real star counts and URLs.
    """
    print("\nFetching GH Archive (real GitHub event stream)...")
    # Try last 3 hours to find one that's available
    records = []
    repo_stars: dict[str, int] = {}
    repo_meta: dict[str, dict] = {}

    for hours_ago in range(2, 8):
        dt = datetime.utcnow() - timedelta(hours=hours_ago)
        url = f"https://data.gharchive.org/{dt.strftime('%Y-%m-%d-%-H')}.json.gz"
        try:
            print(f"  Trying {url} ...")
            resp = requests.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            content = b""
            for chunk in resp.iter_content(chunk_size=65536):
                content += chunk
                if len(content) > 30 * 1024 * 1024:  # stop at 30MB
                    break
            print(f"  Downloaded {len(content)/1024/1024:.1f} MB")
            lines = gzip.decompress(content).decode("utf-8", errors="ignore").split("\n")
            for line in lines:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("type") == "WatchEvent":
                        repo = event.get("repo", {})
                        name = repo.get("name", "")
                        if name:
                            repo_stars[name] = repo_stars.get(name, 0) + 1
                            if name not in repo_meta:
                                repo_meta[name] = {
                                    "url": f"https://github.com/{name}",
                                    "actor": event.get("actor", {}).get("login", ""),
                                    "created_at": event.get("created_at", _now())[:19].replace("T", " "),
                                }
                except Exception:
                    continue
            break
        except Exception as e:
            print(f"  GH Archive error: {e}")
            continue

    if not repo_stars:
        print("  GH Archive unavailable — skipping")
        return []

    # Sort by star velocity, assign domains heuristically
    domain_keywords = {d: q.lower().split() for d, q in DOMAIN_QUERIES.items()}
    top_repos = sorted(repo_stars.items(), key=lambda x: x[1], reverse=True)

    for name, star_count in top_repos[:max_records]:
        name_lower = name.lower()
        domain = "Trending Open-Source"  # default
        for d, kws in domain_keywords.items():
            if any(kw in name_lower for kw in kws[:3]):
                domain = d
                break
        meta = repo_meta.get(name, {})
        records.append({
            "id": _id("gharchive", name),
            "source": "github", "record_type": "repo", "data_source": "live",
            "title": name,
            "description": f"Trending on GitHub — received {star_count} stars in one hour. Real-time GH Archive event data.",
            "url": meta.get("url", f"https://github.com/{name}"),
            "domain": domain, "language": "",
            "tags": json.dumps(["trending", "gharchive"]),
            "stars": star_count * 200,  # rough total stars estimate from velocity
            "forks": 0, "contributors": 0, "open_issues": 0, "good_first_issues": 0,
            "comments": 0, "upvotes": 0,
            "activity_score": min(star_count / 50, 1.0),
            "growth_rate": float(star_count),
            "created_at": meta.get("created_at", _now()),
            "updated_at": _now(),
        })

    print(f"  GH Archive: {len(records)} trending repos extracted")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_records = []
    seen_ids: set[str] = set()

    def add(recs):
        for r in recs:
            if r["id"] not in seen_ids and r.get("title"):
                seen_ids.add(r["id"])
                all_records.append(r)

    print("=" * 60)
    print("EngageIQ — Real Dataset Builder")
    print("=" * 60)

    # 1. GitHub repos (real API)
    print("\n[1/4] GitHub Repos (real API search)...")
    for domain in DOMAINS:
        recs = fetch_github_repos_real(domain, pages=3 if GITHUB_TOKEN else 1)
        add(recs)
        print(f"  {domain}: {len(recs)} repos")

    # 2. GitHub Issues (GFI)
    print("\n[2/4] GitHub Issues — good first issues (real API)...")
    for domain in DOMAINS:
        recs = fetch_github_issues_real(domain, pages=2 if GITHUB_TOKEN else 1)
        add(recs)
        print(f"  {domain}: {len(recs)} issues")

    # 3. Reddit (requires OAuth2 credentials)
    print("\n[3/4] Reddit...")
    if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
        for domain in DOMAINS:
            recs = fetch_reddit_real(domain, n=50)
            add(recs)
            print(f"  {domain}: {len(recs)} posts")
    else:
        print("  Skipped — Reddit API requires OAuth2. Set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET to enable.")

    # 4. Hacker News (real Firebase API)
    print("\n[4/4] Hacker News (real Firebase API)...")
    for domain in DOMAINS:
        recs = fetch_hn_real(domain, n=50)
        add(recs)
        print(f"  {domain}: {len(recs)} stories")

    # 5. GH Archive (real trending stream)
    add(fetch_gharchive(max_records=2000))

    real_df = pd.DataFrame(all_records)
    print(f"\nReal API records collected: {len(real_df)}")

    real_df = real_df.drop_duplicates(subset=["id"])
    real_df.to_csv(str(OUT_CSV), index=False)

    print("\n" + "=" * 60)
    print(f"Saved {len(real_df)} records to {OUT_CSV}")
    print(f"  data_source breakdown: {real_df['data_source'].value_counts().to_dict()}")
    print(f"  record_type breakdown: {real_df['record_type'].value_counts().to_dict()}")
    print(f"  source breakdown:      {real_df['source'].value_counts().to_dict()}")
    print(f"  domains covered:       {real_df['domain'].nunique()}")
    print("=" * 60)

    # Delete cached embeddings so they recompute from new data
    for cache_file in [DATA_DIR / "embeddings.npy", DATA_DIR / "embedding_ids.npy",
                       DATA_DIR / "engageiq.db"]:
        if cache_file.exists():
            cache_file.unlink()
            print(f"Cleared cache: {cache_file.name}")

    print("\nDone! Restart the app to load the new dataset.")


if __name__ == "__main__":
    main()
