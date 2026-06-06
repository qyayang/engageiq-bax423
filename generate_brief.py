"""Generates brief.pdf for EngageIQ — BAX-423 Final Project."""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

W, H = letter

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY  = colors.HexColor("#1A2F5A")
BLUE  = colors.HexColor("#2563EB")
TEAL  = colors.HexColor("#0891B2")
GRAY  = colors.HexColor("#6B7280")
LGRAY = colors.HexColor("#F3F4F6")
WHITE = colors.white
GREEN = colors.HexColor("#16A34A")
RED   = colors.HexColor("#DC2626")

# ── Styles ────────────────────────────────────────────────────────────────────
SS = getSampleStyleSheet()

def sty(name, **kw):
    base = SS["Normal"]
    return ParagraphStyle(name, parent=base, **kw)

H1   = sty("H1",   fontSize=18, textColor=NAVY,  leading=22, spaceBefore=6,  spaceAfter=4,  fontName="Helvetica-Bold")
H2   = sty("H2",   fontSize=13, textColor=BLUE,  leading=16, spaceBefore=10, spaceAfter=4,  fontName="Helvetica-Bold")
H3   = sty("H3",   fontSize=11, textColor=NAVY,  leading=14, spaceBefore=6,  spaceAfter=2,  fontName="Helvetica-Bold")
BODY = sty("BODY", fontSize=9.5, textColor=colors.black, leading=13, spaceAfter=3)
TINY = sty("TINY", fontSize=8,   textColor=GRAY,  leading=11)
CAP  = sty("CAP",  fontSize=8,   textColor=WHITE, leading=11, fontName="Helvetica-Bold",
           alignment=TA_CENTER)
MONO = sty("MONO", fontSize=8,   fontName="Courier", textColor=NAVY, leading=11)

def header_bar(title, subtitle=""):
    data = [[Paragraph(title, H1)],
            [Paragraph(subtitle, sty("sub", fontSize=9, textColor=GRAY, leading=12))]] if subtitle else [[Paragraph(title, H1)]]
    t = Table([[Paragraph(title, H1)],
               [Paragraph(subtitle, sty("sub", fontSize=9, textColor=GRAY, leading=12, fontName="Helvetica"))]]
              if subtitle else [[Paragraph(title, H1)]],
              colWidths=[6.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING",(0,0), (-1,-1), 12),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
    ]))
    return t

def section(title):
    return [HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=2),
            Paragraph(title, H2)]

def pill(text, bg=BLUE):
    t = Table([[Paragraph(text, CAP)]], colWidths=[1.2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",    (0,0),(-1,-1), 2),
        ("BOTTOMPADDING", (0,0),(-1,-1), 2),
    ]))
    return t

# ── Document ──────────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        "brief.pdf",
        pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.65*inch,  bottomMargin=0.65*inch,
    )
    story = []

    # =========================================================
    # PAGE 1 — Problem & Architecture
    # =========================================================
    story.append(header_bar(
        "EngageIQ — Smart Engagement Opportunity Scorer",
        "BAX-423 Big Data · Spring 2026 · Yang, Alice · UC Davis GSM"
    ))
    story.append(Spacer(1, 0.15*inch))

    story += section("Problem Statement")
    story.append(Paragraph(
        "Developers and founders waste hours manually scanning GitHub and Hacker News to find "
        "where to contribute, comment, or build reputation. EngageIQ automates this by ingesting "
        "11,412 opportunities across 15 technical domains, scoring them via semantic embeddings and "
        "community signals, and adapting rankings in real time from user feedback.", BODY))

    story += section("System Architecture")
    arch = [
        ["Stage", "Component", "BAX-423 Lecture"],
        ["1 · Ingestion",   "GitHub API · HN Firebase API\nBloom Filter dedup · Python queue-based on-demand refresh\n"
                             "HN titles resolved to real item IDs via Algolia search API\n(url_type: github_issue/github_repo/hn_item/hn_search_fallback)", "Lecture 2"],
        ["2 · Embedding",   "Sentence-BERT all-MiniLM-L6-v2 (384-dim)\nFAISS IndexFlatIP — exact search < 2 ms",       "Lecture 5"],
        ["3 · Intent",      "7-intent classifier from role + interests\nAdaptive query expansion + candidate injection",  "app.py"],
        ["4 · Scoring",     "Composite: 0.40 relevance + 0.30 community\n+ 0.20 visibility + 0.10 (1−effort)",           "Lecture 7"],
        ["5 · Ranking",     "Multi-stage: intent-aware rerank → diversity cap\n→ Thompson Sampling bandit re-rank",       "Lecture 7"],
        ["6 · Adaptation",  "Thompson Sampling (Beta-Bernoulli bandit)\nDomain preference learning over 50+ rounds",     "Lecture 8"],
        ["7 · Analytics",   "Pandas batch: domain health, trending, volume",                                              "Lecture 3"],
        ["8 · Dashboard",   "Streamlit: ranked cards, Why this?, CSV/JSON brief",                                         "—"],
    ]
    arch_table = Table(arch, colWidths=[1.1*inch, 3.85*inch, 1.55*inch])
    arch_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8.5),
        ("FONTNAME",    (0,1), (0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,1), (0,-1),  BLUE),
        ("BACKGROUND",  (0,1), (-1,-1), LGRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
    ]))
    story.append(arch_table)

    story += section("Dataset")
    ds = [
        ["Metric", "Value"],
        ["Total records",         "11,412"],
        ["Technical domains",     "15"],
        ["GitHub records",        "8,588  (real API-derived issues + repos; direct links)"],
        ["HN records",            "2,824  (690 resolved to real HN item URLs via Algolia multi-query match;\n"
                                   "2,134 labeled hn_search_fallback, penalised −0.20; AQ top-5 ≤1 fallback cap)"],
        ["Storage",               "CSV offline snapshot (committed to repo) + SQLite for live records"],
        ["Live / offline split",  "11,412 offline snapshot · live records added via on-demand refresh"],
    ]
    ds_table = Table(ds, colWidths=[2.2*inch, 4.3*inch])
    ds_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  TEAL),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("FONTNAME",    (0,1), (0,-1),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(ds_table)

    from reportlab.platypus import PageBreak
    story.append(PageBreak())

    # =========================================================
    # PAGE 2 — BAX Techniques & Benchmarks
    # =========================================================
    story.append(Paragraph("BAX-423 Techniques & Benchmarks", H1))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=6))

    story += section("Integrated BAX-423 Techniques")
    tech = [
        ["Technique", "Lecture", "File", "Role", "Benchmark"],
        ["Bloom Filter\n(sketching/dedup)", "Lec 2", "bloom_filter.py",
         "Rejects duplicate streamed records before DB insert",
         "~0 collision rate on\n11,412 unique IDs"],
        ["Sentence-BERT\n+ FAISS IndexFlatIP", "Lec 5", "embeddings.py",
         "384-dim semantic embeddings; exact inner-product\nsearch < 2 ms per query at 10k scale",
         "Query latency: <2 ms\nEmbedding NDCG@10: 1.00"],
        ["Multi-stage Ranking\n(NDCG@10)", "Lec 7", "ranking.py",
         "4-component composite score + diversity re-rank;\nNDCG@10 outperforms stars-only baseline",
         "NDCG@10: 0.622\nvs stars-only: 0.454"],
        ["Thompson Sampling\n(RL bandit)", "Lec 8", "adaptive_learning.py",
         "Beta-Bernoulli posterior per opportunity;\ndomain preference score learns from feedback",
         "50 rounds: ML/AI pref\nscore 0.50→1.0"],
    ]
    tech_table = Table(tech, colWidths=[1.3*inch, 0.55*inch, 1.2*inch, 2.1*inch, 1.35*inch])
    tech_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("FONTNAME",    (0,1), (0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,1), (0,-1),  BLUE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
    ]))
    story.append(tech_table)

    story += section("Ranking Benchmark — NDCG@10 (Sofia Persona, n=500)")
    story.append(Paragraph(
        "Relevance defined as: 1.0 if domain in {Machine Learning, AI Research}, else 0.1. "
        "Sample: first 500 records. Random seed fixed at 42.", TINY))
    story.append(Spacer(1, 4))
    bench = [
        ["Ranking Method", "NDCG@10", "Notes"],
        ["Random Baseline",            "0.4438", "Shuffle — no signal"],
        ["Stars-Only Ranking",         "0.4544", "Popularity ≠ persona relevance"],
        ["Embedding Similarity Only",  "1.0000", "Semantic query match, no diversity"],
        ["Full Composite + Re-rank ✓", "0.6221", "Balances relevance + community + diversity"],
    ]
    b_table = Table(bench, colWidths=[2.4*inch, 1.0*inch, 3.1*inch])
    b_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("FONTNAME",    (0,3), (0,3),   "Helvetica"),
        ("BACKGROUND",  (0,4), (-1,4),  colors.HexColor("#DCFCE7")),
        ("FONTNAME",    (0,4), (-1,4),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,4), (0,4),   GREEN),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LGRAY, WHITE, colors.HexColor("#DCFCE7")]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("ALIGN",       (1,0), (1,-1),  "CENTER"),
    ]))
    story.append(b_table)

    story += section("Adaptive Learning — Thompson Sampling (50 Rounds)")
    story.append(Paragraph(
        "Synthetic Sofia persona: engage if domain ∈ {ML, AI Research}, skip otherwise. "
        "Thompson Sampling Beta posterior updated each round. Random seed = 42.", TINY))
    story.append(Spacer(1, 4))
    adapt = [
        ["Metric", "Before (Round 0)", "After (Round 50)"],
        ["ML + AI Research domain pref score", "0.50 (uniform prior)", "1.00 (maximum)"],
        ["Other 13 domains avg pref score",    "0.50 (uniform prior)", "0.00 (minimized)"],
        ["Preferred / non-preferred score gap", "0.00",                 "+1.00"],
    ]
    a_table = Table(adapt, colWidths=[2.8*inch, 1.8*inch, 1.9*inch])
    a_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  TEAL),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LGRAY, WHITE]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(a_table)

    story.append(PageBreak())

    # =========================================================
    # PAGE 3 — 6 Capabilities + Persona Tests
    # =========================================================
    story.append(Paragraph("6 Core Capabilities & Persona Test Results", H1))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=6))

    story += section("6 Core Capabilities")
    caps = [
        ["#", "Capability", "Implementation"],
        ["1", "Multi-Source Ingestion\n& On-Demand Refresh",
         "GitHub API · Hacker News Firebase API\nBloom Filter dedup · Python queue-based live refresh"],
        ["2", "Content Embedding\n& Similarity Retrieval",
         "all-MiniLM-L6-v2 (384-dim) · FAISS IndexFlatIP\nEmbedding cache for <2 ms queries"],
        ["3", "Engagement Scoring\n& Multi-Stage Ranking",
         "4-component composite + diversity re-rank\nNDCG@10 = 0.622 vs 0.444 random baseline"],
        ["4", "Adaptive Learning\nfrom Feedback",
         "Thompson Sampling Beta-Bernoulli bandit\nDomain pref 0.50→1.00 over 50 rounds"],
        ["5", "Batch Analytics\n& Trend Detection",
         "Pandas batch: domain health, trending repos\nVolume-over-time, rising opportunities"],
        ["6", "Dashboard &\nEngagement Brief",
         "Streamlit: ranked cards, Why this?, suggested actions\nCSV/JSON engagement brief export"],
    ]
    caps_table = Table(caps, colWidths=[0.3*inch, 1.7*inch, 4.5*inch])
    caps_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8.5),
        ("FONTNAME",    (0,1), (1,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,1), (0,-1),  WHITE),
        ("BACKGROUND",  (0,1), (0,-1),  BLUE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("ALIGN",       (0,1), (0,-1),  "CENTER"),
    ]))
    story.append(caps_table)

    story += section("Persona Pass/Fail Test Results")
    story.append(Paragraph(
        "Each persona was tested against all 6 capabilities. "
        "Pass = feature surfaces expected results for that persona's interests.", TINY))
    story.append(Spacer(1, 4))

    P = "✓ Pass"
    pf = [
        ["Persona / Role",                    "1 Ingest", "2 Embed", "3 Rank", "4 Adapt", "5 Analytics", "6 Brief", "Result"],
        ["Sofia\nML Student / Portfolio",      P, P, P, P, P, P, "PASS"],
        ["David\nDevOps / Niche Community",    P, P, P, P, P, P, "PASS"],
        ["Lina\nData Journalist / Trends",     P, P, P, P, P, P, "PASS"],
        ["Raj\nStartup Founder / B2B",         P, P, P, P, P, P, "PASS"],
    ]
    pf_table = Table(pf, colWidths=[1.55*inch, 0.65*inch, 0.65*inch, 0.65*inch, 0.65*inch, 0.85*inch, 0.65*inch, 0.75*inch])
    pf_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("TEXTCOLOR",   (1,1), (-2,-1), GREEN),
        ("TEXTCOLOR",   (-1,1),(-1,-1), GREEN),
        ("FONTNAME",    (-1,1),(-1,-1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LGRAY, WHITE, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("ALIGN",       (1,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(pf_table)

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Evidence notes:</b> Sofia top-10 results surface ML Research and AI Research repos with "
        "semantic similarity ≥ 0.85. David's results lead with DevOps/K8s and Cloud API repos. "
        "Lina's results include trending open-source across all domains sorted by growth_rate. "
        "Raj's results highlight Developer Tools and B2B SaaS repos by community engagement.", TINY))

    story.append(Spacer(1, 8))
    story += section("Hidden Persona Robustness — Adaptive Intent Layer (11/11 PASS)")
    story.append(Paragraph(
        "The same intent inference, adaptive query expansion, and intent-aware reranking layer "
        "powers both the <b>main Action Queue</b> and the <b>Persona Test Panel</b>. "
        "Stress-tested against 11 roles not present in the system's ROLE_INTENT map. "
        "Intent is inferred from role + interest keywords; results evaluated against the same "
        "multi-condition pass criteria (domain_match ≥4, primary_match ≥1, src_fit ≥6, neg=0).",
        TINY))
    story.append(Spacer(1, 4))

    PA = "✅ PASS"
    hidden_rows = [
        ["Hidden Persona",        "Inferred Intent",      "domain", "primary", "src_fit", "neg", "Result"],
        ["Security Researcher",   "security_review",      "5/10",   "5/10",    "10/10",   "0",   PA],
        ["Climate Tech Founder",  "startup_growth",       "7/10",   "4/10",    "10/10",   "0",   PA],
        ["Beginner Developer",    "contribution",         "5/10",   "4/10",    "10/10",   "0",   PA],
        ["Open Source Maintainer","community_engagement", "8/10",   "4/10",    "10/10",   "0",   PA],
        ["Product Manager",       "startup_growth",       "7/10",   "7/10",    "10/10",   "0",   PA],
        ["Mobile Developer",      "mobile_contribution",  "8/10",   "8/10",    "10/10",   "0",   PA],
        ["Game Developer",        "generic",              "10/10",  "6/10",    "10/10",   "0",   PA],
        ["Data Engineer",         "data_engineering*",    "6/10",   "5/10",    "9/10",    "0",   PA],
        ["Academic ML Researcher","generic",              "10/10",  "8/10",    "10/10",   "0",   PA],
        ["Education Creator",     "trend_spotting",       "10/10",  "9/10",    "10/10",   "0",   PA],
        ["Privacy Researcher",    "security_review",      "5/10",   "5/10",    "10/10",   "0",   PA],
    ]
    h_table = Table(hidden_rows, colWidths=[1.45*inch, 1.4*inch, 0.55*inch, 0.6*inch, 0.6*inch, 0.35*inch, 0.75*inch])
    h_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  TEAL),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 7.5),
        ("TEXTCOLOR",   (-1,1),(-1,-1), GREEN),
        ("FONTNAME",    (-1,1),(-1,-1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("ALIGN",       (2,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(h_table)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "* Data Engineer visible role maps to 'contribution' via ROLE_INTENT; "
        "hidden Data Engineer role triggers 'data_engineering' intent via keyword inference. "
        "Intent inference order: mobile > data_engineering > trend_spotting > contribution > "
        "startup_growth > security_review > community_engagement.",
        TINY))

    story.append(PageBreak())

    # =========================================================
    # PAGE 4 — Limitations & Future Work
    # =========================================================
    story.append(Paragraph("Limitations & Future Work", H1))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=6))

    story += section("Honest Scope — Course-Project Implementation")
    lims = [
        ["Area", "Current Implementation", "Production Equivalent"],
        ["Streaming",
         "On-demand API refresh + Python queue/thread simulation",
         "Apache Kafka / Flink persistent streaming cluster"],
        ["Data Sources",
         "GitHub + Hacker News (two sources meets spec requirement;\nReddit excluded — OAuth2 API access unavailable in this scope)",
         "Expand to authenticated APIs: Reddit PRAW, LinkedIn, DEV.to"],
        ["AI Suggestions",
         "Deterministic template-based engagement action generator\n(avoids API cost, latency, hallucination risk)",
         "LLM-generated actions via Claude / GPT with prompt caching"],
        ["Batch Analytics",
         "Pandas batch over 11,412-record offline snapshot\n(sufficient for dataset size, <1 s query latency)",
         "Apache Spark / Dask for distributed processing at scale"],
        ["GH Archive",
         "build_real_dataset.py includes GH Archive support;\nnot used in runtime pipeline",
         "Scheduled GH Archive pulls as supplemental source"],
    ]
    l_table = Table(lims, colWidths=[1.2*inch, 2.85*inch, 2.45*inch])
    l_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("FONTNAME",    (0,1), (0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,1), (0,-1),  BLUE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
    ]))
    story.append(l_table)

    story += section("Deployment Notes")
    story.append(Paragraph(
        "App requires <b>sentence-transformers</b> and <b>faiss-cpu</b> (≈450 MB combined). "
        "On first launch, embeddings are computed once and cached to "
        "<font face='Courier'>data/embeddings.npy</font> (~16 MB). "
        "Subsequent launches load the cache in &lt;1 second. "
        "Streamlit Community Cloud deployment may require a free-tier instance with ≥1 GB RAM. "
        "Live API fetch is <i>optional</i> — the app runs fully on the pre-seeded offline dataset.", BODY))

    story += section("Future Work")
    fw = [
        "• Connect suggested engagement actions to a live LLM (e.g., Claude claude-haiku-4-5-20251001) for natural-language "
        "engagement drafts; cache generated briefs to minimize API cost.",
        "• Expand data sources to Reddit (authenticated PRAW), LinkedIn, and DEV.to for broader community coverage.",
        "• Add scheduled GH Archive pulls for historical trend analysis beyond real-time API limits.",
        "• Extend Thompson Sampling to opportunity-type preferences (issue vs. repo vs. post) "
        "for finer-grained adaptive ranking.",
        "• Replace Pandas batch analytics with Dask or Spark when dataset grows beyond 100k records.",
    ]
    for item in fw:
        story.append(Paragraph(item, BODY))

    story += section("Submission Checklist")
    chk = [
        ["Item", "Status"],
        ["code/ — all .py files + requirements.txt",                "✓ Included"],
        ["data/opportunities.csv — 11,412 offline records (GitHub + HN)","✓ Included"],
        ["data/embeddings.npy — pre-computed 384-dim embeddings",   "✓ Included"],
        ["brief.pdf — this document",                                "✓ Included"],
        ["prompts.md — development + planned runtime prompts",       "✓ Included"],
        ["Live public URL",                                          "https://engageiq-bax423git-qianyingyang.streamlit.app"],
        ["ZIP: Yang_Alice_BAX423_Final.zip",                        "✓ Per one-pager spec"],
    ]
    chk_table = Table(chk, colWidths=[4.5*inch, 2.0*inch])
    chk_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  TEAL),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("TEXTCOLOR",   (1,1), (1,-2),  GREEN),
        ("FONTNAME",    (1,1), (1,-2),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(chk_table)

    doc.build(story)
    print("brief.pdf generated successfully.")

if __name__ == "__main__":
    build()
