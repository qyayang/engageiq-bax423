"""Generates brief.pdf for EngageIQ — BAX-423 Final Project. Black & white only."""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

BLACK  = colors.black
WHITE  = colors.white
LGRAY  = colors.HexColor("#CCCCCC")
LLGRAY = colors.HexColor("#F2F2F2")

SS = getSampleStyleSheet()

def sty(name, **kw):
    return ParagraphStyle(name, parent=SS["Normal"], **kw)

TITLE = sty("TITLE", fontSize=15, textColor=BLACK, leading=19, spaceAfter=2,  fontName="Helvetica-Bold")
H2    = sty("H2",    fontSize=11, textColor=BLACK, leading=14, spaceBefore=8, spaceAfter=2, fontName="Helvetica-Bold")
BODY  = sty("BODY",  fontSize=9,  textColor=BLACK, leading=13, spaceAfter=2)
SMALL = sty("SMALL", fontSize=8,  textColor=BLACK, leading=11, spaceAfter=1)
BOLD  = sty("BOLD",  fontSize=9,  textColor=BLACK, leading=13, fontName="Helvetica-Bold")
TH    = sty("TH",    fontSize=9,  textColor=BLACK, leading=12, fontName="Helvetica-Bold", alignment=TA_CENTER)
TC    = sty("TC",    fontSize=8.5,textColor=BLACK, leading=12)
TC_C  = sty("TC_C",  fontSize=8.5,textColor=BLACK, leading=12, alignment=TA_CENTER)
TC_B  = sty("TC_B",  fontSize=8.5,textColor=BLACK, leading=12, fontName="Helvetica-Bold")

def p(text, style=None):
    """Wrap text in a Paragraph for safe table-cell rendering."""
    return Paragraph(text, style or TC)

def pc(text):
    return Paragraph(text, TC_C)

def pb(text):
    return Paragraph(text, TC_B)

def section(title):
    return [
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=0.5, color=BLACK, spaceAfter=1),
        Paragraph(title, H2),
        Spacer(1, 2),
    ]

BASE_TS = TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  LGRAY),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LLGRAY]),
    ("BOX",           (0, 0), (-1, -1), 0.5, BLACK),
    ("INNERGRID",     (0, 0), (-1, -1), 0.3, BLACK),
    ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING",   (0, 0), (-1, -1), 5),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
])

def build():
    doc = SimpleDocTemplate(
        "brief.pdf",
        pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.65*inch,  bottomMargin=0.65*inch,
    )
    story = []

    # ── PAGE 1 ───────────────────────────────────────────────────────────────
    story.append(Paragraph("EngageIQ — Smart Engagement Opportunity Scorer", TITLE))
    story.append(Paragraph("BAX-423 Big Data · Spring 2026 · Yang, Alice · UC Davis GSM", SMALL))
    story.append(Spacer(1, 8))

    story += section("Problem Statement")
    story.append(Paragraph(
        "Developers and founders waste hours manually scanning GitHub and Hacker News to find "
        "where to contribute, comment, or build reputation. EngageIQ automates this by ingesting "
        "10,046 opportunities across 15 technical domains, scoring them via semantic embeddings "
        "and community signals, and adapting rankings in real time from user feedback.", BODY))

    story += section("System Architecture")
    arch_data = [
        [p("Stage", TH), p("Component", TH), p("Lecture", TH)],
        [pb("1 · Ingestion"),
         p("GitHub API + Algolia/HN API. Bloom Filter dedup. Python queue-based on-demand refresh. "
           "Real HN stories fetched by domain keyword queries (url_type: hn_item, 0 fallbacks)."),
         pc("Lec 2")],
        [pb("2 · Embedding"),
         p("Sentence-BERT all-MiniLM-L6-v2 (384-dim). FAISS IndexFlatIP exact search < 2 ms."),
         pc("Lec 5")],
        [pb("3 · Intent"),
         p("7-intent classifier from role + interests. Adaptive query expansion + candidate injection."),
         pc("app.py")],
        [pb("4 · Scoring"),
         p("Composite: 0.40 relevance + 0.30 community + 0.20 visibility + 0.10 (1-effort)."),
         pc("Lec 7")],
        [pb("5 · Ranking"),
         p("Multi-stage: intent-aware rerank + per-domain diversity cap + Thompson Sampling rerank."),
         pc("Lec 7/8")],
        [pb("6 · Adaptation"),
         p("Thompson Sampling Beta-Bernoulli bandit. Domain preference learning over 50+ rounds."),
         pc("Lec 8")],
        [pb("7 · Analytics"),
         p("Pandas batch: domain health, trending repos, volume-over-time, rising opportunities."),
         pc("Lec 3")],
        [pb("8 · Dashboard"),
         p("Streamlit: ranked cards, Why this?, suggested actions, CSV/JSON export."),
         pc("—")],
    ]
    t = Table(arch_data, colWidths=[1.15*inch, 4.7*inch, 1.15*inch])
    t.setStyle(BASE_TS)
    story.append(t)

    story += section("Dataset")
    ds_data = [
        [p("Metric", TH), p("Value", TH)],
        [pb("Total records"),        p("10,046")],
        [pb("Technical domains"),    p("15")],
        [pb("GitHub records"),       p("8,588 — real API-derived issues + repos; direct github.com links")],
        [pb("HN records"),           p("1,458 — real Algolia/HN API stories; direct news.ycombinator.com/item?id= URLs; 0 fallbacks")],
        [pb("Storage"),              p("CSV offline snapshot (committed to repo) + SQLite for live records")],
        [pb("Live / offline split"), p("10,046 offline snapshot; live records added via on-demand API refresh")],
    ]
    t = Table(ds_data, colWidths=[1.8*inch, 5.2*inch])
    t.setStyle(BASE_TS)
    story.append(t)

    story.append(PageBreak())

    # ── PAGE 2 ───────────────────────────────────────────────────────────────
    story.append(Paragraph("BAX-423 Techniques & Benchmarks", TITLE))
    story.append(Spacer(1, 4))

    story += section("Integrated BAX-423 Techniques")
    tech_data = [
        [p("Technique", TH), p("Lec", TH), p("File", TH), p("Role in System", TH), p("Benchmark", TH)],
        [pb("Bloom Filter\n(sketching/dedup)"),
         pc("2"), p("bloom_filter.py"),
         p("Rejects duplicate streamed records before DB insert."),
         p("~0 collision rate on 10,046 unique IDs")],
        [pb("Sentence-BERT\n+ FAISS IndexFlatIP"),
         pc("5"), p("embeddings.py"),
         p("384-dim semantic embeddings. Exact inner-product search at 10k scale."),
         p("Query latency: <2 ms\nNDCG@10 (emb only): 0.49")],
        [pb("Multi-stage\nRanking (NDCG)"),
         pc("7"), p("ranking.py"),
         p("Composite score + diversity rerank. NDCG@10 outperforms stars-only by 55%."),
         p("NDCG@10: 0.2425\nvs stars-only: 0.1569")],
        [pb("Thompson Sampling\n(RL bandit)"),
         pc("8"), p("adaptive_learning.py"),
         p("Beta-Bernoulli posterior per opportunity. Domain preference learns from feedback."),
         p("50 rounds: ML/AI pref\n0.50 -> 1.0")],
    ]
    t = Table(tech_data, colWidths=[1.15*inch, 0.35*inch, 1.15*inch, 2.45*inch, 1.9*inch])
    t.setStyle(BASE_TS)
    story.append(t)

    story += section("Ranking Benchmark — NDCG@10  (Sofia Persona, n=500, seed=42)")
    story.append(Paragraph(
        "Graded relevance: in-domain records scored 0–1 by stars (40%) + activity (30%) + "
        "community health (20%) + GFI bonus (20%). Higher = better (max 1.0). "
        "Random stratified sample, seed=42.", SMALL))
    story.append(Spacer(1, 3))
    bench_data = [
        [p("Ranking Method", TH), p("NDCG@10", TH), p("Notes", TH)],
        [p("Embedding Similarity Only"),  pc("0.4915"), p("Strong domain retrieval for aligned queries")],
        [pb("Full Composite + Re-rank *"), pc("0.2425"), pb("Composite scoring + diversity; beats stars-only by 55%")],
        [p("Stars-Only Ranking"),          pc("0.1569"), p("Popularity does not equal persona relevance")],
        [p("Random Baseline"),             pc("0.1036"), p("No signal — lower bound")],
    ]
    t = Table(bench_data, colWidths=[2.4*inch, 0.9*inch, 3.7*inch])
    t.setStyle(BASE_TS)
    story.append(t)
    story.append(Paragraph(
        "* Benchmark isolates composite scoring. In the live app, Full Composite also applies "
        "intent-specific GFI/diversity reranking on top of embedding retrieval — complementary stages.", SMALL))

    story += section("Adaptive Learning — Thompson Sampling (50 Rounds)")
    story.append(Paragraph(
        "Synthetic Sofia persona: engage if domain in {ML, AI Research}, skip otherwise. "
        "Beta posterior updated each round. Seed=42.", SMALL))
    story.append(Spacer(1, 3))
    adapt_data = [
        [p("Metric", TH),                              p("Before (Round 0)", TH), p("After (Round 50)", TH)],
        [p("ML + AI Research domain pref score"),      p("0.50 (uniform prior)"), p("1.00 (maximum)")],
        [p("Other 13 domains avg pref score"),         p("0.50 (uniform prior)"), p("0.00 (minimized)")],
        [p("Preferred / non-preferred score gap"),     p("0.00"),                  p("+1.00")],
    ]
    t = Table(adapt_data, colWidths=[3.0*inch, 2.0*inch, 2.0*inch])
    t.setStyle(BASE_TS)
    story.append(t)

    story.append(PageBreak())

    # ── PAGE 3 ───────────────────────────────────────────────────────────────
    story.append(Paragraph("6 Core Capabilities & Persona Test Results", TITLE))
    story.append(Spacer(1, 4))

    story += section("6 Core Capabilities")
    caps_data = [
        [p("#", TH), p("Capability", TH), p("Implementation", TH)],
        [pc("1"), pb("Multi-Source Ingestion\n& On-Demand Refresh"),
         p("GitHub API + Hacker News API. Bloom Filter dedup. Python queue-based live refresh.")],
        [pc("2"), pb("Content Embedding\n& Similarity Retrieval"),
         p("all-MiniLM-L6-v2 (384-dim). FAISS IndexFlatIP. Embedding cache for <2 ms queries.")],
        [pc("3"), pb("Engagement Scoring\n& Multi-Stage Ranking"),
         p("4-component composite + diversity rerank. NDCG@10 = 0.2425 vs 0.1036 random baseline.")],
        [pc("4"), pb("Adaptive Learning\nfrom Feedback"),
         p("Thompson Sampling Beta-Bernoulli bandit. Domain pref 0.50 -> 1.00 over 50 rounds.")],
        [pc("5"), pb("Batch Analytics\n& Trend Detection"),
         p("Pandas batch: domain health, trending repos, volume-over-time, rising opportunities.")],
        [pc("6"), pb("Dashboard &\nEngagement Brief"),
         p("Streamlit: ranked cards, Why this?, suggested actions. CSV/JSON export.")],
    ]
    t = Table(caps_data, colWidths=[0.3*inch, 1.65*inch, 5.05*inch])
    t.setStyle(BASE_TS)
    story.append(t)

    story += section("Persona Pass/Fail Test Results")
    story.append(Paragraph(
        "Each persona tested against all 6 capabilities. "
        "Pass = feature surfaces expected results for that persona's interests.", SMALL))
    story.append(Spacer(1, 3))
    pf_data = [
        [p("Persona / Role", TH), pc("1"), pc("2"), pc("3"), pc("4"), pc("5"), pc("6"), p("Result", TH)],
        [p("Sofia — ML Student"),       pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pb("PASS")],
        [p("David — DevOps Engineer"),  pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pb("PASS")],
        [p("Lina — Data Journalist"),   pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pb("PASS")],
        [p("Raj — Startup Founder"),    pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pc("Pass"), pb("PASS")],
    ]
    t = Table(pf_data, colWidths=[1.7*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.7*inch])
    ts = TableStyle(BASE_TS.getCommands())
    ts.add("ALIGN", (1, 0), (-1, -1), "CENTER")
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Evidence: Sofia top-10 surfaces ML/AI Research repos (gfi=6, domain_match=8). "
        "David leads with DevOps/K8s + Cloud API repos (infra_kw=4, domain_match=9). "
        "Lina includes trending open-source sorted by growth_rate (avg_trend=0.69, hn=10). "
        "Raj highlights Developer Tools and B2B SaaS (api=6, domain_match=10).", SMALL))

    story += section("Hidden Persona Robustness — Adaptive Intent Layer (11/11 PASS)")
    story.append(Paragraph(
        "The same intent inference, adaptive query expansion, and intent-aware reranking layer "
        "powers both the main Action Queue and the Persona Test Panel. "
        "Stress-tested against 11 roles not in ROLE_INTENT map — intent inferred from role + "
        "interest keywords. Pass criteria: domain_match >= 4, primary_match >= 1, "
        "src_fit >= 6, neg = 0.", SMALL))
    story.append(Spacer(1, 3))
    hidden_data = [
        [p("Hidden Persona", TH), p("Inferred Intent", TH), pc("domain"), pc("primary"), pc("src_fit"), pc("neg"), p("Result", TH)],
        [p("Security Researcher"),    p("security_review"),      pc("5/10"), pc("5/10"), pc("10/10"), pc("0"), pb("PASS")],
        [p("Climate Tech Founder"),   p("startup_growth"),       pc("7/10"), pc("4/10"), pc("10/10"), pc("0"), pb("PASS")],
        [p("Beginner Developer"),     p("contribution"),         pc("5/10"), pc("4/10"), pc("10/10"), pc("0"), pb("PASS")],
        [p("Open Source Maintainer"), p("community_engagement"), pc("8/10"), pc("4/10"), pc("10/10"), pc("0"), pb("PASS")],
        [p("Product Manager"),        p("startup_growth"),       pc("7/10"), pc("7/10"), pc("10/10"), pc("0"), pb("PASS")],
        [p("Mobile Developer"),       p("mobile_contribution"),  pc("8/10"), pc("8/10"), pc("10/10"), pc("0"), pb("PASS")],
        [p("Game Developer"),         p("generic"),              pc("10/10"),pc("6/10"), pc("10/10"), pc("0"), pb("PASS")],
        [p("Data Engineer"),          p("data_engineering"),     pc("6/10"), pc("5/10"), pc("9/10"),  pc("0"), pb("PASS")],
        [p("Academic ML Researcher"), p("generic"),              pc("10/10"),pc("8/10"), pc("10/10"), pc("0"), pb("PASS")],
        [p("Education Creator"),      p("trend_spotting"),       pc("10/10"),pc("9/10"), pc("10/10"), pc("0"), pb("PASS")],
        [p("Privacy Researcher"),     p("security_review"),      pc("5/10"), pc("5/10"), pc("10/10"), pc("0"), pb("PASS")],
    ]
    t = Table(hidden_data, colWidths=[1.5*inch, 1.5*inch, 0.58*inch, 0.62*inch, 0.62*inch, 0.38*inch, 0.8*inch])
    ts = TableStyle(BASE_TS.getCommands())
    ts.add("ALIGN", (2, 0), (-1, -1), "CENTER")
    t.setStyle(ts)
    story.append(t)

    story.append(PageBreak())

    # ── PAGE 4 ───────────────────────────────────────────────────────────────
    story.append(Paragraph("Limitations, Future Work & Submission Checklist", TITLE))
    story.append(Spacer(1, 4))

    story += section("Honest Scope — Course-Project Implementation")
    lims_data = [
        [p("Area", TH), p("Current Implementation", TH), p("Production Equivalent", TH)],
        [pb("Streaming"),
         p("On-demand API refresh + Python queue/thread simulation."),
         p("Apache Kafka / Flink persistent streaming cluster.")],
        [pb("Data Sources"),
         p("GitHub + Hacker News. Reddit excluded — OAuth2 unavailable in scope."),
         p("Expand to Reddit PRAW, LinkedIn, DEV.to.")],
        [pb("AI Suggestions"),
         p("Deterministic template-based engagement action generator."),
         p("LLM-generated actions via Claude / GPT with prompt caching.")],
        [pb("Batch Analytics"),
         p("Pandas batch over 10,046-record offline snapshot (<1 s latency)."),
         p("Apache Spark / Dask for distributed processing at scale.")],
        [pb("GH Archive"),
         p("build_real_dataset.py includes GH Archive support; not used in runtime."),
         p("Scheduled GH Archive pulls as supplemental source.")],
    ]
    t = Table(lims_data, colWidths=[1.1*inch, 2.95*inch, 2.95*inch])
    t.setStyle(BASE_TS)
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
        "On first launch, embeddings are computed once and cached to data/embeddings.npy (~16 MB). "
        "Subsequent launches load the cache in < 1 second. "
        "Live API fetch is optional — the app runs fully on the pre-seeded offline dataset.", BODY))

    story += section("Submission Checklist")
    chk_data = [
        [p("Item", TH), p("Status", TH)],
        [p("code/ — all .py files + requirements.txt"),                  pb("Included")],
        [p("data/opportunities.csv — 10,046 offline records"),           pb("Included")],
        [p("data/embeddings.npy — pre-computed 384-dim embeddings"),     pb("Included")],
        [p("brief.pdf — this document"),                                  pb("Included")],
        [p("prompts.md — development + planned runtime prompts"),         pb("Included")],
        [p("Live public URL"), p("https://engageiq-bax423git-qianyingyang.streamlit.app")],
        [p("GitHub Repo"),     p("https://github.com/qyayang/engageiq-bax423")],
        [p("ZIP: Yang_Alice_BAX423_Final.zip"),                           pb("Per one-pager spec")],
    ]
    t = Table(chk_data, colWidths=[3.5*inch, 3.5*inch])
    t.setStyle(BASE_TS)
    story.append(t)

    doc.build(story)
    print("brief.pdf generated successfully.")


if __name__ == "__main__":
    build()
