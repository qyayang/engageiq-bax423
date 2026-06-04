"""
Live API data collectors.
Fetches fresh opportunities from GitHub REST API v3, Reddit (PRAW), and Hacker News.
Deduplicates against the Bloom filter before inserting into the database.
"""
import hashlib
import json
import os
import threading
import queue
import time
from datetime import datetime
from typing import Optional

import requests

from bloom_filter import BloomFilter

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = "EngageIQ/1.0"

_HEADERS_GH = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    _HEADERS_GH["Authorization"] = f"Bearer {GITHUB_TOKEN}"

# Domain → GitHub search query mapping
DOMAIN_QUERIES = {
    "Machine Learning": "machine learning PyTorch neural network",
    "DevOps/K8s": "kubernetes helm terraform ci-cd",
    "Trending Open-Source": "trending stars:>1000",
    "Developer Tools": "cli developer tools productivity",
    "Cybersecurity": "security vulnerability scanner OWASP",
    "Frontend (React/Web)": "react component library typescript",
    "B2B SaaS": "saas multi-tenant billing",
    "Blockchain": "smart-contract ethereum solidity defi",
    "Python Data Eng": "data pipeline spark airflow dbt",
    "GameDev (C++)": "game engine graphics opengl vulkan",
    "AI Research": "large language model alignment RLHF",
    "Embedded Systems (C/RTOS)": "embedded rtos freertos microcontroller",
    "Cloud APIs": "aws gcp azure serverless cloud",
    "Mobile Dev (iOS/Flutter)": "flutter ios swift mobile",
    "Beginner Coding": "good-first-issues beginner tutorial",
}

DOMAIN_SUBREDDITS = {
    "Machine Learning": ["MachineLearning", "learnmachinelearning"],
    "DevOps/K8s": ["devops", "kubernetes"],
    "Trending Open-Source": ["opensource", "programming"],
    "Developer Tools": ["programming", "webdev"],
    "Cybersecurity": ["netsec", "cybersecurity"],
    "Frontend (React/Web)": ["reactjs", "webdev"],
    "B2B SaaS": ["SaaS", "startups"],
    "Blockchain": ["ethereum", "defi"],
    "Python Data Eng": ["dataengineering", "Python"],
    "GameDev (C++)": ["gamedev", "cpp"],
    "AI Research": ["MachineLearning", "artificial"],
    "Embedded Systems (C/RTOS)": ["embedded", "RTOS"],
    "Cloud APIs": ["aws", "googlecloud"],
    "Mobile Dev (iOS/Flutter)": ["FlutterDev", "iOSProgramming"],
    "Beginner Coding": ["learnprogramming", "learnpython"],
}


def _make_id(source: str, raw_id: str) -> str:
    return hashlib.md5(f"{source}:{raw_id}".encode()).hexdigest()[:16]


def fetch_github_issues(domain: str, per_page: int = 30) -> list[dict]:
    """
    Fetches real GitHub issues with 'good first issue' or 'help wanted' labels.
    Issue-level recommendations are more actionable than repo-level.
    """
    query = DOMAIN_QUERIES.get(domain, domain)
    search_q = f"{query} label:\"good first issue\" state:open"
    url = "https://api.github.com/search/issues"
    params = {"q": search_q, "sort": "created", "order": "desc", "per_page": per_page}
    try:
        resp = requests.get(url, headers=_HEADERS_GH, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception:
        return []

    records = []
    for item in items:
        repo_url = item.get("repository_url", "")
        repo_name = repo_url.split("repos/")[-1] if "repos/" in repo_url else ""
        labels = [lb["name"] for lb in item.get("labels", [])]
        comments = item.get("comments", 0)
        record = {
            "id": _make_id("github_issue", str(item["id"])),
            "source": "github",
            "title": f"[Issue] {item.get('title', '')}",
            "description": (
                f"GitHub issue in {repo_name}. Labels: {', '.join(labels)}. "
                f"{comments} comments. {(item.get('body') or '')[:200]}"
            ),
            "url": item.get("html_url", ""),
            "domain": domain,
            "language": "",
            "tags": json.dumps(labels[:5]),
            "stars": 0,
            "forks": 0,
            "contributors": 0,
            "open_issues": 1,
            "good_first_issues": 1,  # always 1 since we searched for GFI
            "comments": comments,
            "upvotes": 0,
            "activity_score": round(min((comments + 1) / 20, 1.0), 4),
            "growth_rate": 0.0,
            "created_at": item.get("created_at", ""),
            "updated_at": item.get("updated_at", ""),
        }
        records.append(record)
    return records


def fetch_github_repos(domain: str, per_page: int = 30) -> list[dict]:
    query = DOMAIN_QUERIES.get(domain, domain)
    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
    }
    try:
        resp = requests.get(url, headers=_HEADERS_GH, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception:
        return []

    records = []
    for item in items:
        record = {
            "id": _make_id("github", str(item["id"])),
            "source": "github",
            "title": item.get("full_name", item.get("name", "")),
            "description": item.get("description") or "",
            "url": item.get("html_url", ""),
            "domain": domain,
            "language": item.get("language") or "",
            "tags": json.dumps(item.get("topics", [])),
            "stars": item.get("stargazers_count", 0),
            "forks": item.get("forks_count", 0),
            "contributors": 0,
            "open_issues": item.get("open_issues_count", 0),
            "good_first_issues": 0,
            "comments": item.get("open_issues_count", 0),
            "upvotes": 0,
            "activity_score": 0.0,
            "growth_rate": 0.0,
            "created_at": item.get("created_at", ""),
            "updated_at": item.get("updated_at", ""),
        }
        # Compute basic activity score
        s = record["stars"]
        f = record["forks"]
        record["activity_score"] = round(min((s + f * 2) / 50000, 1.0), 4)
        records.append(record)
    return records


def fetch_hn_stories(domain: str, n: int = 20) -> list[dict]:
    """Fetches top HN stories and filters by domain keywords."""
    keywords = DOMAIN_QUERIES.get(domain, domain).lower().split()
    try:
        top_ids = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json", timeout=8
        ).json()[:200]
    except Exception:
        return []

    records = []
    for story_id in top_ids[:100]:
        if len(records) >= n:
            break
        try:
            item = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                timeout=5,
            ).json()
            if not item or item.get("type") not in ("story", "ask", "show"):
                continue
            title = item.get("title", "").lower()
            if not any(kw[:4] in title for kw in keywords[:3]):
                continue
            score = item.get("score", 0)
            n_comments = item.get("descendants", 0)
            record = {
                "id": _make_id("hackernews", str(story_id)),
                "source": "hackernews",
                "title": item.get("title", ""),
                "description": f"HN story — score: {score}, {n_comments} comments. {item.get('url', '')}",
                "url": item.get("url") or f"https://news.ycombinator.com/item?id={story_id}",
                "domain": domain,
                "language": "",
                "tags": json.dumps(keywords[:3]),
                "stars": 0,
                "forks": 0,
                "contributors": 0,
                "open_issues": 0,
                "good_first_issues": 0,
                "comments": n_comments,
                "upvotes": score,
                "activity_score": round(min((score + n_comments) / 1000, 1.0), 4),
                "growth_rate": round(score / 10, 2),
                "created_at": datetime.fromtimestamp(item.get("time", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            records.append(record)
        except Exception:
            continue
    return records


def fetch_reddit_posts(domain: str, n: int = 20) -> list[dict]:
    """Fetches Reddit posts via the public JSON API (no auth required for public subreddits)."""
    subreddits = DOMAIN_SUBREDDITS.get(domain, ["programming"])
    records = []
    for sub in subreddits[:2]:
        if len(records) >= n:
            break
        try:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit={n}"
            resp = requests.get(
                url,
                headers={"User-Agent": REDDIT_USER_AGENT},
                timeout=8,
            )
            resp.raise_for_status()
            posts = resp.json().get("data", {}).get("children", [])
        except Exception:
            continue

        for post in posts:
            data = post.get("data", {})
            if data.get("stickied"):
                continue
            score = data.get("score", 0)
            n_comments = data.get("num_comments", 0)
            record = {
                "id": _make_id("reddit", data.get("id", "")),
                "source": "reddit",
                "title": data.get("title", ""),
                "description": (data.get("selftext") or "")[:300] or f"Reddit r/{sub}: {data.get('title', '')}",
                "url": f"https://reddit.com{data.get('permalink', '')}",
                "domain": domain,
                "language": "",
                "tags": json.dumps([sub]),
                "stars": 0,
                "forks": 0,
                "contributors": 0,
                "open_issues": 0,
                "good_first_issues": 0,
                "comments": n_comments,
                "upvotes": score,
                "activity_score": round(min((score + n_comments * 5) / 10000, 1.0), 4),
                "growth_rate": round(score / 100, 2),
                "created_at": datetime.fromtimestamp(data.get("created_utc", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            records.append(record)
    return records


class StreamingIngester:
    """
    Simulated streaming ingester — producer/consumer pattern.
    Mimics Kafka producer → consumer pipeline using Python queue.
    Fetches new records from live APIs and streams them through the pipeline.
    """

    def __init__(self, bloom_filter: BloomFilter, on_new_record=None):
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._bloom = bloom_filter
        self._on_new_record = on_new_record or (lambda r: None)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.records_ingested = 0
        self.duplicates_blocked = 0

    def start(self, domains: list[str], interval_sec: int = 60):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._producer_loop,
            args=(domains, interval_sec),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def drain(self) -> list[dict]:
        """Drain all queued records (consumer side)."""
        results = []
        while not self._queue.empty():
            try:
                results.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _producer_loop(self, domains: list[str], interval_sec: int):
        while not self._stop.is_set():
            for domain in domains:
                if self._stop.is_set():
                    break
                new_records = (
                    fetch_hn_stories(domain, n=5)
                    + fetch_reddit_posts(domain, n=5)
                    + fetch_github_issues(domain, per_page=5)
                )
                for record in new_records:
                    uid = record.get("url", record.get("id", ""))
                    if self._bloom.contains(uid):
                        self.duplicates_blocked += 1
                        continue
                    self._bloom.add(uid)
                    try:
                        self._queue.put_nowait(record)
                        self.records_ingested += 1
                        self._on_new_record(record)
                    except queue.Full:
                        pass
            self._stop.wait(timeout=interval_sec)
