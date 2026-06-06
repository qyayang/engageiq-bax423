"""Generates brief.pdf for EngageIQ — BAX-423 Final Project. Black & white only."""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

BLACK  = colors.black
WHITE  = colors.white
LGRAY  = colors.HexColor("#DDDDDD")
LLGRAY = colors.HexColor("#F5F5F5")

SS = getSampleStyleSheet()

def sty(name, **kw):
    return ParagraphStyle(name, parent=SS["Normal"], **kw)

H1   = sty("H1",   fontSize=16, textColor=BLACK, leading=20, spaceBefore=0, spaceAfter=4,  fontName="Helvetica-Bold")
H2   = sty("H2",   fontSize=11, textColor=BLACK, leading=14, spaceBefore=8, spaceAfter=3,  fontName="Helvetica-Bold")
BODY = sty("BODY", fontSize=9,  textColor=BLACK, leading=13, spaceAfter=3)
TINY = sty("TINY", fontSize=8,  textColor=BLACK, leading=11)
MONO = sty("MONO", fontSize=8,  fontName="Courier", textColor=BLACK, leading=11)

def section(title):
    return [
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=0.5, color=BLACK, spaceAfter=2),
        Paragraph(title, H2),
    ]

def table_style(col_widths, header_bold=True):
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  LGRAY),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  BLACK),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS",(0,1), (-1, -1), [WHITE, LLGRAY]),
        ("GRID",         (0, 0), (-1, -1), 0.4, BLACK),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ])


def build():
    doc = SimpleDocTemplate(
        "brief.pdf",
        pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.65*inch,  bottomMargin=0.65*inch,
    )
    story = []
    W = 7.0 * inch  # usable width

    # ── PAGE 1: Problem, Architecture, Dataset ──────────────────────────────
    story.append(Paragraph("EngageIQ — Smart Engagement Opportunity Scorer", H1))
    story.append(Paragraph("BAX-423 Big Data · Spring 2026 · Yang, Alice · UC Davis GSM", TINY))
    story.append(Spacer(1, 6))

    story += section("Problem Statement")
    story.append(Paragraph(
        "Developers and founders waste hours manually scanning GitHub and Hacker News to find "
        "where to contribute, comment, or build reputation. EngageIQ automates this by ingesting "
        "10,046 opportunities across 15 technical domains, scoring them via semantic embeddings and "
        "community signals, and adapting rankings in real time from user feedback.", BODY))

    story += section("System Architecture")
    arch = [
        ["Stage", "Component", "Lecture"],
        ["1 · Ingestion",
         "GitHub API + Algolia/HN API; Bloom Filter dedup; Python queue refresh\n"
         "Real HN stories via domain keyword queries (hn_item URLs, 0 fallbacks)",
         "Lec 2"],
        ["2 · Embedding",
         "Sentence-BERT all-MiniLM-L6-v2 (384-dim); FAISS IndexFlatIP <2 ms",
         "Lec 5"],
        ["3 · Intent",
         "7-intent classifier from role + interests; adaptive query expansion + candidate injection",
         "app.py"],
        ["4 · Scoring",
         "Composite: 0.40 relevance + 0.30 community + 0.20 visibility + 0.10 (1-effort)",
         "Lec 7"],
        ["5 · Ranking",
         "Multi-stage: intent-aware rerank + diversity cap + Thompson Sampling bandit",
         "Lec 7/8"],
        ["6 · Adaptation",
         "Thompson Sampling Beta-Bernoulli bandit; domain preference learning over 50+ rounds",
         "Lec 8"],
        ["7 · Analytics",
         "Pandas batch: domain health, trending repos, volume-over-time",
         "Lec 3"],
        ["8 · Dashboard",
         "Streamlit: ranked cards, Why this?, suggested actions, CSV/JSON export",
         "—"],
    ]
    t = Table(arch, colWidths=[1.1*inch, 4.4*inch, 1.5*inch])
    t.setStyle(table_style(None))
    story.append(t)

    story += section("Dataset")
    ds = [
        ["Metric", "Value"],
        ["Total records",        "10,046"],
        ["Technical domains",    "15"],
        ["GitHub records",       "8,588 — real API-derived issues + repos; direct github.com links"],
        ["HN records",           "1,458 — real Algolia/HN API stories; direct news.ycombinator.com/item?id= URLs; 0 fallbacks"],
        ["Storage",              "CSV offline snapshot (committed to repo) + SQLite for live records"],
        ["Live / offline split", "10,046 offline snapshot; live records added via on-demand refresh"],
    ]
    t = Table(ds, colWidths=[1.8*inch, 5.2*inch])
    t.setStyle(table_style(None))
    story.append(t)

    story.append(PageBreak())

    # ── PAGE 2: BAX Techniques & Benchmarks ────────────────────────────────
    story.append(Paragraph("BAX-423 Techniques & Benchmarks", H1))
    story.append(Spacer(1, 4))

    story += section("Integrated BAX-423 Techniques")
    tech = [
        ["Technique", "Lec", "File", "Role", "Benchmark"],
        ["Bloom Filter\n(sketching/dedup)",    "2", "bloom_filter.py",
         "Rejects duplicate streamed records before DB insert",
         "~0 collision rate on 10,046 unique IDs"],
        ["Sentence-BERT\n+ FAISS IndexFlatIP", "5", "embeddings.py",
         "384-dim semantic embeddings; exact inner-product search <2 ms at 10k scale",
         "Query latency: <2 ms\nNDCG@10 emb-only: 0.49"],
        ["Multi-stage Ranking\n(NDCG@10)",     "7", "ranking.py",
         "Composite score + diversity rerank; beats stars-only by 55%",
         "NDCG@10: 0.2425\nvs stars-only: 0.1569"],
        ["Thompson Sampling\n(RL bandit)",      "8", "adaptive_learning.py",
         "Beta-Bernoulli posterior per opportunity; domain pref learns from feedback",
         "50 rounds: ML/AI pref 0.50 -> 1.0"],
    ]
    t = Table(tech, colWidths=[1.25*inch, 0.35*inch, 1.2*inch, 2.3*inch, 1.9*inch])
    t.setStyle(table_style(None))
    story.append(t)

    story += section("Ranking Benchmark — NDCG@10 (Sofia Persona, n=500, seed=42)")
    story.append(Paragraph(
        "Quality-aligned graded relevance: in-domain records scored 0-1 by stars (40%) + "
        "activity (30%) + community health (20%) + GFI bonus (20%). "
        "Higher NDCG@10 = better (max = 1.0). Random stratified sample, seed=42.", TINY))
    story.append(Spacer(1, 3))
    bench = [
        ["Ranking Method",               "NDCG@10", "Notes"],
        ["Embedding Similarity Only",    "0.4915",  "Strong domain retrieval for aligned queries"],
        ["Full Composite + Re-rank *",   "0.2425",  "Composite scoring + diversity; beats stars-only by 55%"],
        ["Stars-Only Ranking",           "0.1569",  "Popularity != persona relevance"],
        ["Random Baseline",              "0.1036",  "No signal — lower bound"],
    ]
    t = Table(bench, colWidths=[2.4*inch, 0.9*inch, 3.7*inch])
    ts = table_style(None)
    ts.add("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold")
    t.setStyle(ts)
    story.append(t)
    story.append(Paragraph(
        "* Benchmark isolates composite scoring from intent-aware reranking. In the live app, "
        "Full Composite also applies intent-specific GFI/diversity reranking — the two stages are complementary.", TINY))

    story += section("Adaptive Learning — Thompson Sampling (50 Rounds)")
    story.append(Paragraph(
        "Synthetic Sofia persona: engage if domain in {ML, AI Research}, skip otherwise. "
        "Thompson Sampling Beta posterior updated each round. Random seed = 42.", TINY))
    story.append(Spacer(1, 3))
    adapt = [
        ["Metric",                                  "Before (Round 0)",   "After (Round 50)"],
        ["ML + AI Research domain pref score",      "0.50 (uniform prior)", "1.00 (maximum)"],
        ["Other 13 domains avg pref score",         "0.50 (uniform prior)", "0.00 (minimized)"],
        ["Preferred / non-preferred score gap",     "0.00",                 "+1.00"],
    ]
    t = Table(adapt, colWidths=[3.0*inch, 2.0*inch, 2.0*inch])
    t.setStyle(table_style(None))
    story.append(t)

    story.append(PageBreak())

    # ── PAGE 3: 6 Capabilities + Persona Tests ──────────────────────────────
    story.append(Paragraph("6 Core Capabilities & Persona Test Results", H1))
    story.append(Spacer(1, 4))

    story += section("6 Core Capabilities")
    caps = [
        ["#", "Capability",                       "Implementation"],
        ["1",  "Multi-Source Ingestion\n& On-Demand Refresh",
         "GitHub API + Hacker News API; Bloom Filter dedup; Python queue-based refresh"],
        ["2",  "Content Embedding\n& Similarity Retrieval",
         "all-MiniLM-L6-v2 (384-dim); FAISS IndexFlatIP; embedding cache for <2 ms queries"],
        ["3",  "Engagement Scoring\n& Multi-Stage Ranking",
         "4-component composite + diversity rerank; NDCG@10 = 0.2425 vs 0.1036 random baseline"],
        ["4",  "Adaptive Learning\nfrom Feedback",
         "Thompson Sampling Beta-Bernoulli bandit; domain pref 0.50->1.00 over 50 rounds"],
        ["5",  "Batch Analytics\n& Trend Detection",
         "Pandas batch: domain health, trending repos, volume-over-time, rising opportunities"],
        ["6",  "Dashboard &\nEngagement Brief",
         "Streamlit: ranked cards, Why this?, suggested actions; CSV/JSON export"],
    ]
    t = Table(caps, colWidths=[0.3*inch, 1.7*inch, 5.0*inch])
    t.setStyle(table_style(None))
    story.append(t)

    story += section("Persona Pass/Fail Test Results")
    story.append(Paragraph(
        "Each persona tested against all 6 capabilities. "
        "Pass = feature surfaces expected results for that persona's interests.", TINY))
    story.append(Spacer(1, 3))
    pf = [
        ["Persona / Role",                 "1 Ingest", "2 Embed", "3 Rank", "4 Adapt", "5 Analytics", "6 Brief", "Result"],
        ["Sofia — ML Student / Portfolio",  "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "PASS"],
        ["David — DevOps / Niche Comm.",    "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "PASS"],
        ["Lina — Data Journalist / Trends", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "PASS"],
        ["Raj — Startup Founder / B2B",     "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "PASS"],
    ]
    t = Table(pf, colWidths=[1.7*inch, 0.65*inch, 0.65*inch, 0.6*inch, 0.65*inch, 0.85*inch, 0.6*inch, 0.7*inch])
    ts = table_style(None)
    ts.add("FONTNAME", (0, 1), (-1, -1), "Helvetica")
    ts.add("ALIGN",    (1, 0), (-1, -1), "CENTER")
    ts.add("FONTNAME", (-1, 1), (-1, -1), "Helvetica-Bold")
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Evidence: Sofia top-10 surfaces ML/AI Research repos (gfi>=6, domain_match=8). "
        "David leads with DevOps/K8s and Cloud API repos (infra_kw=4, domain_match=9). "
        "Lina includes trending open-source sorted by growth_rate (avg_trend=0.69). "
        "Raj highlights Developer Tools and B2B SaaS by community engagement (api=6, dm=10).", TINY))

    story += section("Hidden Persona Robustness — Adaptive Intent Layer (11/11 PASS)")
    story.append(Paragraph(
        "The same intent inference, adaptive query expansion, and intent-aware reranking layer "
        "powers both the main Action Queue and the Persona Test Panel. "
        "Stress-tested against 11 roles not in the system's ROLE_INTENT map — "
        "intent inferred from role + interest keywords; evaluated against multi-condition pass criteria "
        "(domain_match>=4, primary_match>=1, src_fit>=6, neg=0).", TINY))
    story.append(Spacer(1, 3))
    hidden = [
        ["Hidden Persona",         "Inferred Intent",      "domain", "primary", "src_fit", "neg", "Result"],
        ["Security Researcher",    "security_review",      "5/10",   "5/10",    "10/10",   "0",   "PASS"],
        ["Climate Tech Founder",   "startup_growth",       "7/10",   "4/10",    "10/10",   "0",   "PASS"],
        ["Beginner Developer",     "contribution",         "5/10",   "4/10",    "10/10",   "0",   "PASS"],
        ["Open Source Maintainer", "community_engagement", "8/10",   "4/10",    "10/10",   "0",   "PASS"],
        ["Product Manager",        "startup_growth",       "7/10",   "7/10",    "10/10",   "0",   "PASS"],
        ["Mobile Developer",       "mobile_contribution",  "8/10",   "8/10",    "10/10",   "0",   "PASS"],
        ["Game Developer",         "generic",              "10/10",  "6/10",    "10/10",   "0",   "PASS"],
        ["Data Engineer",          "data_engineering",     "6/10",   "5/10",    "9/10",    "0",   "PASS"],
        ["Academic ML Researcher", "generic",              "10/10",  "8/10",    "10/10",   "0",   "PASS"],
        ["Education Creator",      "trend_spotting",       "10/10",  "9/10",    "10/10",   "0",   "PASS"],
        ["Privacy Researcher",     "security_review",      "5/10",   "5/10",    "10/10",   "0",   "PASS"],
    ]
    t = Table(hidden, colWidths=[1.5*inch, 1.5*inch, 0.55*inch, 0.6*inch, 0.6*inch, 0.35*inch, 0.8*inch])
    ts = table_style(None)
    ts.add("ALIGN", (2, 0), (-1, -1), "CENTER")
    ts.add("FONTNAME", (-1, 1), (-1, -1), "Helvetica-Bold")
    t.setStyle(ts)
    story.append(t)

    story.append(PageBreak())

    # ── PAGE 4: Limitations, Future Work, Checklist ─────────────────────────
    story.append(Paragraph("Limitations, Future Work & Submission Checklist", H1))
    story.append(Spacer(1, 4))

    story += section("Honest Scope — Course-Project Implementation")
    lims = [
        ["Area",          "Current Implementation",                              "Production Equivalent"],
        ["Streaming",
         "On-demand API refresh + Python queue/thread simulation",
         "Apache Kafka / Flink persistent streaming cluster"],
        ["Data Sources",
         "GitHub + Hacker News (two sources meets spec;\nReddit excluded — OAuth2 unavailable in scope)",
         "Expand to Reddit PRAW, LinkedIn, DEV.to"],
        ["AI Suggestions",
         "Deterministic template-based engagement action generator\n(avoids API cost and hallucination risk)",
         "LLM-generated actions via Claude / GPT with prompt caching"],
        ["Batch Analytics",
         "Pandas batch over 10,046-record offline snapshot\n(<1 s query latency at this scale)",
         "Apache Spark / Dask for distributed processing at scale"],
        ["GH Archive",
         "build_real_dataset.py includes GH Archive support;\nnot used in runtime pipeline",
         "Scheduled GH Archive pulls as supplemental source"],
    ]
    t = Table(lims, colWidths=[1.1*inch, 3.0*inch, 2.9*inch])
    t.setStyle(table_style(None))
    story.append(t)

    story += section("Future Work")
    for item in [
        "Connect suggested engagement actions to a live LLM (e.g., Claude Haiku) for natural-language drafts; cache generated briefs to minimize API cost.",
        "Expand data sources to Reddit (authenticated PRAW), LinkedIn, and DEV.to for broader community coverage.",
        "Add scheduled GH Archive pulls for historical trend analysis beyond real-time API limits.",
        "Extend Thompson Sampling to opportunity-type preferences (issue vs. repo vs. post) for finer-grained adaptive ranking.",
        "Replace Pandas batch analytics with Dask or Spark when dataset grows beyond 100k records.",
    ]:
        story.append(Paragraph("• " + item, BODY))

    story += section("Deployment Notes")
    story.append(Paragraph(
        "App requires sentence-transformers and faiss-cpu (~450 MB combined). "
        "On first launch embeddings are computed once and cached to data/embeddings.npy (~16 MB). "
        "Subsequent launches load the cache in <1 second. "
        "Live API fetch is optional — the app runs fully on the pre-seeded offline dataset.", BODY))

    story += section("Submission Checklist")
    chk = [
        ["Item",                                                      "Status"],
        ["code/ — all .py files + requirements.txt",                  "Included"],
        ["data/opportunities.csv — 10,046 offline records",           "Included"],
        ["data/embeddings.npy — pre-computed 384-dim embeddings",     "Included"],
        ["brief.pdf — this document",                                  "Included"],
        ["prompts.md — development + planned runtime prompts",         "Included"],
        ["Live public URL",   "https://engageiq-bax423git-qianyingyang.streamlit.app"],
        ["GitHub Repo",       "https://github.com/qyayang/engageiq-bax423"],
        ["ZIP: Yang_Alice_BAX423_Final.zip",                           "Per one-pager spec"],
    ]
    t = Table(chk, colWidths=[4.0*inch, 3.0*inch])
    ts = table_style(None)
    ts.add("FONTNAME", (1, 1), (1, -1), "Helvetica-Bold")
    t.setStyle(ts)
    story.append(t)

    doc.build(story)
    print("brief.pdf generated successfully.")


if __name__ == "__main__":
    build()
