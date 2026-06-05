"""
Offline dataset generator — produces ≥10,000 records across 15 technical domains.
Sources: GitHub repos/issues + Hacker News stories (no Reddit).
Run once: python generate_offline_data.py
"""
import hashlib
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_CSV = DATA_DIR / "opportunities.csv"

# ── Domain configuration ──────────────────────────────────────────────────────
DOMAIN_CFG = {
    "Machine Learning": {
        "languages": ["Python", "Jupyter Notebook", "Python"],
        "subreddits": ["MachineLearning", "learnmachinelearning", "MLQuestions", "deeplearning"],
        "stars_mu": 8.5, "stars_sigma": 2.2,
        "contrib_range": (5, 600), "gfi_prob": 0.38,
        "keywords": ["neural network", "deep learning", "PyTorch", "gradient descent",
                     "transformer", "LLM", "model training", "loss function",
                     "backpropagation", "convolutional", "BERT", "GPT"],
        "repo_names": [
            ("awesome-ml-papers", "Curated list of machine learning research papers with implementations"),
            ("pytorch-lightning-examples", "Clean PyTorch Lightning examples for common ML tasks including classification, regression, and sequence modeling"),
            ("sklearn-extensions", "Extended scikit-learn utilities for preprocessing, feature selection, and model evaluation with AutoML support"),
            ("transformer-from-scratch", "Step-by-step implementation of attention mechanisms and transformer architecture in pure NumPy and PyTorch"),
            ("ml-interview-prep", "Comprehensive ML interview preparation with coding challenges, theory questions, and system design case studies"),
            ("deep-rl-playground", "Deep reinforcement learning algorithms including DQN, PPO, and A3C with OpenAI Gym environments"),
            ("feature-store-lite", "Lightweight feature store for ML pipelines with caching, versioning, and real-time serving support"),
            ("model-compression-toolkit", "Neural network pruning, quantization, and knowledge distillation toolkit for production deployment"),
            ("mlops-pipeline-template", "Production-ready MLOps pipeline template with experiment tracking, model registry, and CI/CD integration"),
            ("nlp-benchmark-suite", "Benchmarking suite for NLP models across tasks: text classification, NER, QA, and summarization"),
            ("federated-learning-framework", "Privacy-preserving federated learning with differential privacy and secure aggregation"),
            ("time-series-ml", "Machine learning toolkit for time-series forecasting, anomaly detection, and classification with LSTM and Prophet"),
            ("explainable-ai-toolkit", "Tools for model interpretability including SHAP, LIME, and attention visualization for transformers"),
            ("data-augmentation-library", "Data augmentation strategies for images, text, and tabular data with configurable pipelines"),
            ("auto-feature-engineering", "Automated feature engineering using genetic programming and deep feature synthesis for tabular ML"),
            ("contrastive-learning-pytorch", "Self-supervised contrastive learning implementations: SimCLR, MoCo, BYOL with benchmarks"),
            ("graph-neural-networks", "PyTorch Geometric tutorials and implementations for node classification, link prediction, and graph generation"),
            ("multimodal-learning-hub", "Multimodal learning combining vision, text, and audio with fusion architectures and zero-shot transfer"),
            ("active-learning-toolkit", "Active learning strategies for efficient annotation with uncertainty sampling and query-by-committee"),
            ("ml-system-design-guide", "Practical guide to designing ML systems at scale: feature pipelines, training infrastructure, and serving"),
        ],
        "reddit_posts": [
            "What's the best beginner-friendly ML project to contribute to on GitHub?",
            "Finally got my first PR merged into a major NLP library — here's what I learned",
            "Resources for understanding transformer architecture from scratch (with code)",
            "How do you stay updated with ML papers without getting overwhelmed?",
            "Monthly Thread: Share your ML projects for feedback and collaboration",
            "Ask ML: Which open-source projects are actively looking for contributors?",
            "I implemented a transformer from scratch — benchmarks vs HuggingFace",
            "Tips for writing good first issues that attract quality contributors",
            "Best practices for ML experiment tracking in 2026",
            "Show r/ML: Built a lightweight BERT fine-tuning framework — feedback welcome",
        ],
        "hn_posts": [
            "Ask HN: How do you find beginner-friendly ML open-source projects?",
            "Show HN: I trained a small language model on 1B tokens, here's what I learned",
            "The state of open-source LLMs in 2026",
            "Why most ML papers don't reproduce: a study of 1000 implementations",
            "Show HN: PyTorch training loop that reduced my iteration time by 3x",
        ],
    },

    "DevOps/K8s": {
        "languages": ["Go", "YAML", "Python", "Shell", "TypeScript"],
        "subreddits": ["devops", "kubernetes", "docker", "terraform", "sre"],
        "stars_mu": 7.8, "stars_sigma": 2.0,
        "contrib_range": (3, 200), "gfi_prob": 0.22,
        "keywords": ["Kubernetes", "container", "Helm chart", "CI/CD", "observability",
                     "Prometheus", "Grafana", "Terraform", "IaC", "operator",
                     "GitOps", "ArgoCD", "service mesh", "Istio"],
        "repo_names": [
            ("k8s-operator-framework", "Kubernetes operator framework for building cloud-native controllers with reconciliation loop patterns"),
            ("helm-chart-library", "Production-ready Helm charts for common microservices patterns with security hardening and resource limits"),
            ("terraform-aws-modules", "Reusable Terraform modules for AWS infrastructure with best practices for VPC, EKS, and RDS"),
            ("gitops-bootstrap", "GitOps bootstrap toolkit using ArgoCD and Flux for declarative cluster management"),
            ("prometheus-alert-rules", "Curated Prometheus alerting rules for Kubernetes workloads, databases, and infrastructure components"),
            ("k8s-debugging-toolkit", "Debugging toolkit for Kubernetes: pod inspection, network diagnostics, and resource analysis"),
            ("ci-cd-pipeline-templates", "Reusable CI/CD pipeline templates for GitHub Actions, GitLab CI, and Jenkins with security scanning"),
            ("istio-service-mesh-guide", "Comprehensive Istio service mesh configuration with traffic management, mTLS, and observability"),
            ("kubernetes-cost-optimizer", "Kubernetes resource optimization tool that analyzes utilization and recommends rightsizing"),
            ("devops-interview-questions", "Comprehensive DevOps interview preparation covering Kubernetes, networking, and system design"),
            ("chaos-engineering-toolkit", "Chaos engineering toolkit for Kubernetes: network faults, pod failures, and latency injection"),
            ("platform-engineering-reference", "Platform engineering reference architecture with developer portals, IDP, and self-service infrastructure"),
            ("observability-stack-k8s", "Complete observability stack on Kubernetes: metrics, logs, traces with OpenTelemetry"),
            ("multi-cluster-manager", "Multi-cluster Kubernetes management with federated deployments and cross-cluster networking"),
            ("docker-security-scanner", "Container image security scanner with CVE detection, SBOM generation, and policy enforcement"),
        ],
        "reddit_posts": [
            "How do you manage Kubernetes cluster upgrades without downtime?",
            "Best tools for monitoring Kubernetes cost and resource efficiency in 2026",
            "We migrated 200 microservices to K8s — lessons learned after 6 months",
            "What's your GitOps workflow? ArgoCD vs Flux comparison after 2 years",
            "How to find good-first-issues in CNCF projects as a DevOps engineer",
            "Debugging Kubernetes networking issues — a practical guide",
            "Show r/devops: Built an open-source Kubernetes cost dashboard",
            "Platform engineering is replacing DevOps — agree or disagree?",
            "Terraform vs Pulumi vs CDK in 2026 — which one do you use?",
            "Thread: contribute to CNCF projects — current openings and mentorship",
        ],
        "hn_posts": [
            "Show HN: Open-source Kubernetes operator that manages 10k clusters",
            "Ask HN: What's the hardest part of running Kubernetes in production?",
            "GitOps is the future of DevOps — a field report from 3 years of practice",
            "We replaced our entire CI/CD stack with GitHub Actions — results",
            "Show HN: A Helm chart linter that caught $50k/month in cloud waste",
        ],
    },

    "Trending Open-Source": {
        "languages": ["Rust", "Go", "TypeScript", "Python", "C++"],
        "subreddits": ["opensource", "programming", "github", "devops"],
        "stars_mu": 9.2, "stars_sigma": 2.5,
        "contrib_range": (10, 800), "gfi_prob": 0.30,
        "keywords": ["trending", "viral", "open-source", "community", "popular",
                     "fast-growing", "emerging", "new release", "starred", "forked",
                     "alternative", "lightweight", "blazing-fast", "zero-dependency"],
        "repo_names": [
            ("fast-json-parser", "Zero-dependency JSON parser written in Rust — 10x faster than serde_json for large payloads"),
            ("ui-component-library", "Accessible, customizable UI component library with React, Vue, and Svelte adapters"),
            ("local-ai-runner", "Run large language models locally on CPU — supports llama, mistral, and phi models"),
            ("open-code-interpreter", "Open-source code interpreter that executes Python, JS, and SQL in sandboxed environments"),
            ("realtime-database-sync", "Real-time database synchronization library that works offline-first with conflict resolution"),
            ("terminal-productivity-suite", "Modern terminal productivity suite: fuzzy finder, session manager, and AI-powered suggestions"),
            ("distributed-cache", "Distributed in-memory cache with Redis-compatible protocol, written in Go for ultra-low latency"),
            ("workflow-automation-engine", "Open-source workflow automation engine — self-hosted n8n alternative with 200+ integrations"),
            ("vector-database-lite", "Lightweight vector database for embeddings with HNSW indexing and Python/TypeScript SDKs"),
            ("open-telemetry-toolkit", "OpenTelemetry toolkit for automatic instrumentation of Python, Go, and Node.js applications"),
            ("markdown-presentation-tool", "Terminal-based presentation tool with Markdown input, live preview, and export to PDF"),
            ("api-gateway-rust", "High-performance API gateway built in Rust with rate limiting, auth, and observability built-in"),
            ("browser-automation-lib", "Browser automation library with natural language instructions — alternative to Selenium"),
            ("private-pastebin", "Self-hosted encrypted pastebin with expiring links and zero-knowledge architecture"),
            ("git-activity-visualizer", "Beautiful terminal visualization of Git repository activity, contributor graphs, and code churn"),
        ],
        "reddit_posts": [
            "This week's fastest-growing open-source repos you should know about",
            "The open-source project that went from 0 to 10k stars in a week — how?",
            "Best open-source alternatives to popular SaaS tools in 2026",
            "How to get your open-source project noticed — lessons from building in public",
            "Monthly: What open-source projects are you excited about this month?",
            "The hidden gems of GitHub — projects with <500 stars that deserve more",
            "Viral GitHub repos and what they have in common",
            "How I turned my side project into a 5k-star open-source tool",
            "Open-source sustainability — how to build a community that lasts",
            "Show r/programming: Launched my tool, went from 0 to 2k stars in 48 hours",
        ],
        "hn_posts": [
            "Ask HN: What open-source project do you wish existed?",
            "Show HN: My side project hit 10k stars — here's the launch story",
            "The open-source tools we rely on that are dangerously underfunded",
            "Show HN: I built a Rust alternative to X, here are the benchmarks",
            "GitHub trending is broken — here's a better way to discover repos",
        ],
    },

    "Developer Tools": {
        "languages": ["TypeScript", "Python", "Go", "Rust", "JavaScript"],
        "subreddits": ["programming", "webdev", "devtools", "productivity"],
        "stars_mu": 8.0, "stars_sigma": 2.1,
        "contrib_range": (3, 300), "gfi_prob": 0.32,
        "keywords": ["CLI", "IDE", "debugger", "linter", "formatter", "profiler",
                     "productivity", "developer experience", "DX", "scaffolding",
                     "code generation", "testing", "benchmarking", "API client"],
        "repo_names": [
            ("smart-cli-framework", "Build beautiful, type-safe CLI applications in Python with auto-generated help and shell completion"),
            ("universal-api-client", "Universal API client with OpenAPI spec parsing, mock generation, and TypeScript code generation"),
            ("code-quality-analyzer", "Static code analysis tool that detects anti-patterns, complexity hotspots, and tech debt"),
            ("env-manager", "Cross-platform environment variable manager with secret scanning, .env validation, and team sync"),
            ("database-schema-migrator", "Schema migration tool for PostgreSQL, MySQL, and SQLite with automatic rollback support"),
            ("test-data-factory", "Generate realistic test data using AI — respects schema constraints and referential integrity"),
            ("performance-profiler-py", "Low-overhead Python performance profiler with flame graphs, memory tracking, and CI integration"),
            ("git-workflow-automation", "Git workflow automation: conventional commits, changelog generation, and semantic versioning"),
            ("api-mock-server", "Instant mock server from OpenAPI specs with realistic response simulation and error injection"),
            ("monorepo-tooling", "Monorepo tooling for TypeScript/Python projects with incremental builds and dependency visualization"),
            ("developer-dashboard", "Self-hosted developer dashboard aggregating PR reviews, CI status, and incident alerts"),
            ("code-review-bot", "AI-powered code review bot that catches bugs, suggests improvements, and enforces style guides"),
            ("dependency-auditor", "Cross-language dependency auditor with CVE tracking, license compliance, and upgrade recommendations"),
            ("local-dev-environment", "Reproducible local development environments using containers — faster than Docker Compose"),
            ("openapi-code-generator", "OpenAPI → typed client SDK generator for Python, TypeScript, Go, and Java with test scaffolding"),
        ],
        "reddit_posts": [
            "What developer tools have changed your workflow the most in the last year?",
            "I built a CLI tool that saves me 2 hours a week — show and tell thread",
            "The best open-source developer tools for API development in 2026",
            "How to make a developer tool people actually want to contribute to",
            "Monthly thread: Share your dev tools and get feedback",
            "What's missing from the current developer tooling ecosystem?",
            "Show r/devtools: Made a smart .env file manager with team sync",
            "Debugging tools that every Python developer should know",
            "The DX gap — why developer experience matters for open-source adoption",
            "Best ways to contribute to developer tooling projects as a newcomer",
        ],
        "hn_posts": [
            "Ask HN: What developer tool do you wish existed but doesn't?",
            "Show HN: A CLI that replaces 5 of my most-used shell scripts",
            "The best dev tools are invisible — thoughts on developer experience",
            "Show HN: We built an open-source alternative to Postman",
            "Ask HN: How do you manage dotfiles across multiple machines?",
        ],
    },

    "Cybersecurity": {
        "languages": ["Python", "Go", "C", "Rust", "Shell"],
        "subreddits": ["netsec", "cybersecurity", "hacking", "AskNetsec"],
        "stars_mu": 7.5, "stars_sigma": 2.3,
        "contrib_range": (2, 150), "gfi_prob": 0.18,
        "keywords": ["vulnerability", "penetration testing", "OWASP", "exploit",
                     "CTF", "threat intelligence", "SIEM", "security scanning",
                     "fuzzing", "reverse engineering", "WAF", "zero-day"],
        "repo_names": [
            ("security-audit-toolkit", "Open-source security audit toolkit for web applications — OWASP Top 10 automated checks"),
            ("network-scanner-go", "Fast network scanner with service fingerprinting, vulnerability detection, and report generation"),
            ("ctf-challenge-collection", "Curated CTF challenges across web, crypto, pwn, and reverse engineering with writeup resources"),
            ("threat-model-generator", "Automated threat modeling for web applications using STRIDE methodology and MITRE ATT&CK"),
            ("secrets-scanner", "Pre-commit secrets scanner that detects API keys, passwords, and tokens across 100+ patterns"),
            ("fuzzing-framework", "Coverage-guided fuzzing framework for C/C++ and Python applications with crash triage"),
            ("security-policy-as-code", "Security policies as code using OPA/Rego for Kubernetes, cloud, and CI/CD enforcement"),
            ("open-soc-platform", "Open-source SOC platform with log ingestion, correlation rules, and incident management"),
            ("red-team-automation", "Red team automation scripts for infrastructure enumeration and privilege escalation testing"),
            ("container-security-bench", "Container security benchmark tool based on CIS Docker Benchmark with auto-remediation"),
            ("osint-toolkit", "OSINT toolkit for information gathering, domain analysis, and social graph mapping"),
            ("zero-trust-blueprint", "Zero-trust network architecture blueprint with reference implementation for cloud environments"),
            ("password-audit-tool", "Password audit tool for testing password policies, common patterns, and breach database checks"),
            ("web-app-firewall-rules", "Community-maintained ModSecurity WAF rules for common attack patterns and bot detection"),
            ("cryptography-playground", "Cryptography playground: implement and break classic and modern ciphers with interactive challenges"),
        ],
        "reddit_posts": [
            "Best open-source security tools every developer should have in their arsenal",
            "How to start contributing to security-focused open-source projects",
            "Weekly CTF thread: challenges, writeups, and team recruitment",
            "Ask netsec: best resources for learning web application security in 2026",
            "I built a secrets scanner that caught 400+ exposed keys in public repos",
            "The OWASP Top 10 hasn't changed much — what does that tell us?",
            "Show r/netsec: Built an open-source WAF rule generator using LLMs",
            "Getting started with bug bounties as a developer — my first 6 months",
            "Why security tools need better UX — a rant and some solutions",
            "How to write good security-related good-first-issues",
        ],
        "hn_posts": [
            "Ask HN: How do you stay current with security vulnerabilities?",
            "Show HN: Open-source SIEM that handles 1M events/sec on a single machine",
            "The state of open-source security tooling in 2026",
            "Show HN: I found 50 API keys in public GitHub repos using this tool",
            "Why static analysis tools miss most security bugs — and what helps",
        ],
    },

    "Frontend (React/Web)": {
        "languages": ["TypeScript", "JavaScript", "CSS", "HTML"],
        "subreddits": ["reactjs", "webdev", "frontend", "javascript"],
        "stars_mu": 8.3, "stars_sigma": 2.2,
        "contrib_range": (5, 500), "gfi_prob": 0.42,
        "keywords": ["React", "component", "hooks", "state management", "CSS",
                     "accessibility", "performance", "TypeScript", "Vite", "Next.js",
                     "animation", "responsive design", "design system"],
        "repo_names": [
            ("accessible-component-kit", "Accessible React component library following WAI-ARIA standards with screen reader support"),
            ("react-performance-toolkit", "React performance optimization toolkit: memoization analyzer, bundle visualizer, render profiler"),
            ("design-token-system", "Cross-framework design token system with automatic dark mode, themes, and Figma sync"),
            ("animation-library-react", "Declarative animation library for React with gesture support, spring physics, and SVG animations"),
            ("state-machine-hooks", "React state machine hooks using XState-inspired patterns for complex UI state management"),
            ("micro-frontend-framework", "Micro-frontend framework for building scalable multi-team web applications with module federation"),
            ("css-in-js-zero-runtime", "Zero-runtime CSS-in-JS solution with TypeScript support and automatic critical CSS extraction"),
            ("web-components-library", "Framework-agnostic web components library compatible with React, Vue, Angular, and vanilla JS"),
            ("form-validation-engine", "Performant form validation engine for React with Zod schema integration and accessible error messages"),
            ("infinite-scroll-virtualized", "Virtualized infinite scroll with dynamic item heights, keyboard navigation, and accessibility"),
            ("color-scheme-generator", "Accessible color scheme generator: WCAG contrast ratios, palette generation, and export to CSS/Tailwind"),
            ("react-testing-utilities", "Testing utilities for React: better queries, async helpers, and mocked API patterns"),
            ("nextjs-starter-enterprise", "Enterprise Next.js starter with auth, i18n, API routes, testing setup, and deployment config"),
            ("frontend-observability", "Frontend observability toolkit: Core Web Vitals monitoring, error tracking, and user session replay"),
            ("headless-data-table", "Headless, accessible data table for React with sorting, filtering, pagination, and virtual scrolling"),
        ],
        "reddit_posts": [
            "What React patterns do you wish more open-source projects used?",
            "Show r/reactjs: Built an accessible table component — looking for contributors",
            "The React ecosystem in 2026 — which libraries survived?",
            "Best first contributions to make in major React projects",
            "How to build a React component library that people actually adopt",
            "CSS-in-JS is dead, long live CSS Modules — a 2026 retrospective",
            "Monthly: Best frontend tools and libraries released this month",
            "Ask r/webdev: How do you handle accessibility in your component library?",
            "TypeScript patterns in React that every developer should know",
            "My first open-source contribution to React — what I learned",
        ],
        "hn_posts": [
            "Ask HN: Is React still the right choice for new projects in 2026?",
            "Show HN: A headless component library with zero CSS dependencies",
            "Web performance in 2026 — what still matters and what's solved",
            "Show HN: Built a Next.js starter that cuts setup time from 2 hours to 5 minutes",
            "The hidden cost of JavaScript frameworks — a performance deep-dive",
        ],
    },

    "B2B SaaS": {
        "languages": ["TypeScript", "Python", "Go", "Ruby"],
        "subreddits": ["SaaS", "startups", "Entrepreneur", "microsaas"],
        "stars_mu": 6.8, "stars_sigma": 2.0,
        "contrib_range": (2, 100), "gfi_prob": 0.20,
        "keywords": ["SaaS", "subscription", "billing", "multi-tenant", "B2B",
                     "customer success", "onboarding", "analytics", "API", "webhook",
                     "enterprise", "pricing", "usage-based", "PLG"],
        "repo_names": [
            ("open-billing-engine", "Open-source billing engine for SaaS — usage-based pricing, metered billing, and Stripe integration"),
            ("saas-analytics-dashboard", "Analytics dashboard template for B2B SaaS with MRR, churn, LTV, and cohort analysis"),
            ("multi-tenant-boilerplate", "Production-ready multi-tenant SaaS boilerplate with row-level security, auth, and billing"),
            ("customer-success-platform", "Open-source customer success platform with health scores, playbooks, and in-app messaging"),
            ("feature-flag-service", "Self-hosted feature flag service with gradual rollouts, A/B testing, and user targeting"),
            ("saas-onboarding-flows", "SaaS onboarding flow library with interactive tours, checklists, and progress tracking"),
            ("webhook-delivery-engine", "Reliable webhook delivery engine with retries, dead letter queue, and delivery monitoring"),
            ("usage-metering-service", "Usage metering service for SaaS products with real-time tracking and billing integration"),
            ("saas-admin-panel", "Admin panel template for B2B SaaS: user management, billing, support tickets, and audit logs"),
            ("api-rate-limiter", "API rate limiting service with tiered plans, burst allowances, and Redis-backed counters"),
            ("product-led-growth-toolkit", "PLG toolkit: viral loops, referral programs, freemium conversion, and growth analytics"),
            ("saas-landing-page-kit", "Conversion-optimized SaaS landing page kit with A/B testing and analytics integration"),
            ("enterprise-sso-integration", "Enterprise SSO integration library supporting SAML, OIDC, and LDAP with JIT provisioning"),
            ("data-export-pipeline", "Self-service data export pipeline for SaaS: scheduled exports, custom formats, and S3 delivery"),
            ("revenue-recognition-engine", "Automated revenue recognition engine for SaaS following ASC 606 / IFRS 15 standards"),
        ],
        "reddit_posts": [
            "What open-source B2B SaaS tools do you use in your stack?",
            "Built a self-hosted billing engine as an alternative to Stripe Billing",
            "How PLG companies think about open-source — a founder perspective",
            "The SaaS tools I wish existed as open-source alternatives",
            "Monthly B2B SaaS show and tell — share what you're building",
            "Ask r/SaaS: What's the hardest part of building multi-tenant architecture?",
            "Show r/SaaS: Open-source feature flag service with 1k stars",
            "Why I open-sourced our internal analytics tool and what happened",
            "Contributing to SaaS tooling open-source — where to start?",
            "Enterprise SaaS integrations — the nightmare and how open-source helps",
        ],
        "hn_posts": [
            "Show HN: Self-hosted Stripe Billing alternative — full source available",
            "Ask HN: What SaaS boilerplate do you recommend in 2026?",
            "Why we open-sourced our B2B analytics platform",
            "Show HN: PLG analytics dashboard I built for my own SaaS",
            "The SaaS graveyard of features nobody uses — a reflection",
        ],
    },

    "Blockchain": {
        "languages": ["Solidity", "Rust", "Go", "TypeScript", "Python"],
        "subreddits": ["ethereum", "solana", "defi", "web3"],
        "stars_mu": 7.2, "stars_sigma": 2.4,
        "contrib_range": (3, 200), "gfi_prob": 0.15,
        "keywords": ["smart contract", "DeFi", "NFT", "blockchain", "Ethereum",
                     "Solana", "Web3", "token", "consensus", "cryptography",
                     "decentralized", "DAO", "Layer 2", "ZK proof"],
        "repo_names": [
            ("defi-protocol-template", "Production-ready DeFi protocol template with AMM, lending pools, and governance contracts"),
            ("solidity-security-patterns", "Solidity security patterns and anti-patterns for smart contract developers with audit checklist"),
            ("zk-proof-playground", "Zero-knowledge proof playground with circom circuits, groth16 proofs, and interactive tutorials"),
            ("blockchain-indexer", "Fast blockchain event indexer supporting Ethereum, Polygon, and Solana with GraphQL API"),
            ("web3-testing-framework", "Web3 testing framework for smart contracts with fork testing, fuzzing, and gas profiling"),
            ("dao-governance-contracts", "DAO governance contracts with quadratic voting, delegation, and time-locked execution"),
            ("token-launch-toolkit", "ERC-20/SPL token launch toolkit with vesting schedules, airdrop mechanics, and tokenomics modeling"),
            ("defi-analytics-dashboard", "DeFi analytics dashboard for tracking TVL, APY, liquidity flows, and protocol health"),
            ("nft-marketplace-contracts", "Gas-optimized NFT marketplace contracts with royalties, batch minting, and on-chain metadata"),
            ("layer2-bridge-contracts", "Layer 2 bridge contracts for asset transfers between Ethereum mainnet and rollups"),
            ("blockchain-data-pipeline", "Blockchain data pipeline for extracting, transforming, and analyzing on-chain events at scale"),
            ("wallet-sdk-typescript", "TypeScript wallet SDK supporting multiple chains with hardware wallet and WalletConnect support"),
            ("cross-chain-protocol", "Cross-chain messaging protocol with attestation, relayer network, and fraud proofs"),
            ("defi-yield-optimizer", "DeFi yield optimizer that automatically moves funds across protocols for maximum APY"),
            ("smart-contract-auditor", "Automated smart contract auditor using static analysis and AI-powered vulnerability detection"),
        ],
        "reddit_posts": [
            "Best resources for learning Solidity and smart contract security in 2026",
            "Show r/ethereum: Built a ZK proof tutorial that's actually beginner-friendly",
            "DeFi protocol auditing — how to get started as a developer contributor",
            "The state of Layer 2 ecosystems — a developer perspective",
            "What open-source blockchain projects are actively accepting contributors?",
            "Ask r/web3: Best frameworks for building dApps in 2026?",
            "I audited 50 DeFi protocols — here's what I found",
            "How to contribute to blockchain infrastructure as a Rust developer",
            "Gas optimization techniques that actually work in 2026",
            "Monthly: New DeFi projects and smart contracts to watch",
        ],
        "hn_posts": [
            "Ask HN: Is blockchain technology finally finding its killer use case?",
            "Show HN: ZK proof system that runs in the browser with WASM",
            "The technical reality of decentralization in 2026",
            "Show HN: Smart contract static analyzer that found bugs in 30% of tested contracts",
            "Why ZK rollups will win the Layer 2 wars — a technical analysis",
        ],
    },

    "Python Data Eng": {
        "languages": ["Python", "SQL", "Scala", "YAML"],
        "subreddits": ["dataengineering", "Python", "apachespark", "dbtlabs"],
        "stars_mu": 7.9, "stars_sigma": 2.0,
        "contrib_range": (5, 300), "gfi_prob": 0.28,
        "keywords": ["data pipeline", "ETL", "PySpark", "Airflow", "dbt",
                     "Kafka", "streaming", "batch processing", "data lake",
                     "data warehouse", "Parquet", "Delta Lake", "Iceberg"],
        "repo_names": [
            ("data-pipeline-framework", "Lightweight data pipeline framework with DAG scheduling, lineage tracking, and great_expectations integration"),
            ("pyspark-utilities", "PySpark utility library with DataFrame helpers, schema validation, and performance benchmarking"),
            ("dbt-models-library", "Reusable dbt model library for common data warehouse patterns: SCD Type 2, fact tables, bridge tables"),
            ("streaming-pipeline-kit", "Streaming data pipeline kit using Kafka and Flink with exactly-once semantics"),
            ("data-quality-framework", "Data quality framework with built-in checks, alerting, and lineage-aware validation"),
            ("delta-lake-tools", "Delta Lake tooling for Python: compaction, vacuum, time travel, and schema evolution utilities"),
            ("data-catalog-lightweight", "Lightweight data catalog for documenting datasets, tracking lineage, and managing metadata"),
            ("airflow-operator-library", "Custom Airflow operators and hooks for 50+ data systems with retry logic and SLA monitoring"),
            ("sql-testing-framework", "SQL testing framework for data warehouse models: unit tests, integration tests, and snapshot testing"),
            ("data-observability-toolkit", "Data observability toolkit: freshness checks, volume anomalies, and distribution drift detection"),
            ("iceberg-python-client", "Python client for Apache Iceberg with catalog integration, partition evolution, and time travel"),
            ("parquet-optimization-guide", "Guide and utilities for optimizing Parquet file layouts for columnar query performance"),
            ("real-time-feature-store", "Real-time feature store for ML with low-latency serving, point-in-time joins, and online/offline sync"),
            ("data-mesh-toolkit", "Data mesh toolkit for implementing domain-oriented data ownership and federated governance"),
            ("lakehouse-benchmark", "Lakehouse benchmark comparing Delta Lake, Iceberg, and Hudi on TPC-DS queries at scale"),
        ],
        "reddit_posts": [
            "How do you structure data pipelines for maintainability in 2026?",
            "PySpark vs Polars — a performance comparison after migrating 50 pipelines",
            "Best practices for contributing to data engineering open-source projects",
            "Show r/dataengineering: Built a lightweight data quality framework",
            "What data engineering tools have you built internally that should be open-sourced?",
            "dbt best practices thread — share your modeling patterns",
            "Ask r/dataeng: how do you handle schema evolution in streaming pipelines?",
            "The modern data stack in 2026 — what survived, what didn't",
            "How I contributed to Apache Airflow and what I learned",
            "Data observability vs data quality — what's the difference and why it matters",
        ],
        "hn_posts": [
            "Ask HN: What's your data engineering stack in 2026?",
            "Show HN: Python data pipeline framework that handles 10TB/day on a laptop",
            "dbt is eating the data warehouse — a field report",
            "Show HN: Data quality tool that caught a $2M billing error",
            "Why streaming data pipelines are harder than they look",
        ],
    },

    "GameDev (C++)": {
        "languages": ["C++", "C", "HLSL", "GLSL", "Python"],
        "subreddits": ["gamedev", "cpp", "opengl", "vulkan"],
        "stars_mu": 7.0, "stars_sigma": 2.3,
        "contrib_range": (2, 150), "gfi_prob": 0.20,
        "keywords": ["game engine", "rendering", "shader", "ECS", "physics",
                     "OpenGL", "Vulkan", "DirectX", "pathfinding", "procedural",
                     "audio engine", "scene graph", "collision detection"],
        "repo_names": [
            ("minimal-game-engine", "Minimal C++ game engine for learning: ECS architecture, OpenGL renderer, and 2D physics"),
            ("vulkan-tutorial-series", "Step-by-step Vulkan tutorial series with complete source code and detailed explanations"),
            ("procedural-terrain-gen", "Procedural terrain generation library using noise functions, erosion simulation, and LOD"),
            ("entity-component-system", "High-performance ECS library in modern C++20 with cache-friendly archetypes and query system"),
            ("game-physics-engine", "Rigid body physics engine with broadphase collision, constraints, and continuous collision detection"),
            ("pathfinding-library-cpp", "Pathfinding library implementing A*, Theta*, JPS, and flow fields for games and robotics"),
            ("audio-synthesis-engine", "Real-time audio synthesis engine with DSP filters, procedural sound, and FMOD integration"),
            ("shader-playground", "Interactive shader playground with live GLSL/HLSL editing, preset effects, and documentation"),
            ("2d-game-framework", "Lightweight 2D game framework in C++ with sprite batching, tilemaps, and input handling"),
            ("game-networking-lib", "Game networking library: UDP reliability, prediction, lag compensation, and anti-cheat utilities"),
            ("level-editor-opengl", "Cross-platform level editor for 2D/3D games with asset pipeline and export to common formats"),
            ("voxel-engine-demo", "Voxel world engine with infinite terrain, dynamic lighting, and chunk streaming"),
            ("game-ai-behaviors", "Game AI behavior toolkit: behavior trees, GOAP, steering behaviors, and influence maps"),
            ("renderer-abstraction-layer", "Graphics API abstraction layer supporting OpenGL, Vulkan, and Metal with minimal overhead"),
            ("cpp-game-math-library", "Header-only C++ game math library: vectors, matrices, quaternions, and geometric primitives"),
        ],
        "reddit_posts": [
            "How do you find beginner-friendly C++ game engine projects to contribute to?",
            "Show r/gamedev: Implemented a minimal Vulkan renderer from scratch",
            "Resources for learning game physics engine development in 2026",
            "The best open-source game engines for learning low-level graphics",
            "Ask r/cpp: What game engine code should every C++ developer read?",
            "I contributed to a major game engine — here's my experience",
            "Monthly: Open-source gamedev tools and engines thread",
            "How to get started with ECS architecture in a small game project",
            "The state of Vulkan vs OpenGL for new projects in 2026",
            "Show r/gamedev: Built a procedural world generator with seed sharing",
        ],
        "hn_posts": [
            "Ask HN: What's the best open-source game engine to learn from?",
            "Show HN: Implemented a Doom-like renderer in 500 lines of C",
            "How game engines handle thousands of objects efficiently",
            "Show HN: A voxel engine that runs in the browser at 60fps",
            "ECS architecture: why every game engine eventually rediscovers it",
        ],
    },

    "AI Research": {
        "languages": ["Python", "CUDA", "C++", "Julia"],
        "subreddits": ["MachineLearning", "artificial", "ArtificialIntelligence", "LocalLLaMA"],
        "stars_mu": 9.0, "stars_sigma": 2.3,
        "contrib_range": (5, 1000), "gfi_prob": 0.25,
        "keywords": ["large language model", "diffusion", "multimodal", "RLHF",
                     "alignment", "interpretability", "scaling laws", "emergent behavior",
                     "fine-tuning", "PEFT", "LoRA", "quantization", "inference"],
        "repo_names": [
            ("llm-training-recipes", "Production LLM training recipes with FSDP, mixed precision, and activation checkpointing"),
            ("alignment-research-toolkit", "Toolkit for AI alignment research: reward modeling, RLHF pipeline, and preference data collection"),
            ("diffusion-model-zoo", "Comprehensive diffusion model implementations: DDPM, DDIM, latent diffusion, and ControlNet"),
            ("interpretability-tools", "Neural network interpretability toolkit: probing classifiers, activation patching, and circuit analysis"),
            ("multimodal-benchmark", "Multimodal AI benchmark covering vision-language, audio-visual, and document understanding tasks"),
            ("efficient-transformers", "Efficient transformer implementations: FlashAttention, sparse attention, and linear attention variants"),
            ("lora-fine-tuning-hub", "LoRA fine-tuning hub with pre-trained adapters, training recipes, and evaluation harness"),
            ("reasoning-evaluation-suite", "Evaluation suite for LLM reasoning: math, code, logic, and multi-step problem solving"),
            ("ai-safety-benchmarks", "AI safety benchmarks measuring robustness, honesty, and harmlessness across model families"),
            ("synthetic-data-generation", "Synthetic training data generation for LLMs using instruction back-translation and self-play"),
            ("model-merging-toolkit", "Model merging toolkit: TIES, DARE, SLERP for combining fine-tuned checkpoints"),
            ("llm-inference-optimizer", "LLM inference optimizer: continuous batching, speculative decoding, and KV cache management"),
            ("reward-model-library", "Reward model library for RLHF with preference data handling and calibration utilities"),
            ("agent-evaluation-harness", "Evaluation harness for LLM agents: tool use, multi-turn reasoning, and environment interaction"),
            ("scaling-experiments", "Reproducible scaling experiments for neural language models following Chinchilla compute budget"),
        ],
        "reddit_posts": [
            "Best repos to study if you want to understand how modern LLMs work",
            "Show r/ML: Reproduced a major paper from scratch — annotated code",
            "Resources for understanding RLHF and alignment techniques",
            "The most impactful open-source AI research releases this month",
            "How to contribute to AI safety research as a software engineer",
            "Ask r/ML: What AI research directions are most underfunded and open?",
            "Monthly: Best AI/ML papers with open-source implementations",
            "The gap between AI research and production — how open-source bridges it",
            "Diffusion models are replacing everything — a technical breakdown",
            "Show r/LocalLLaMA: Running Llama 4 locally — benchmarks and tricks",
        ],
        "hn_posts": [
            "Ask HN: What AI research areas do you think are most important in 2026?",
            "Show HN: Open-source RLHF pipeline that reproduced InstructGPT results",
            "Why AI interpretability research matters more than ever",
            "Show HN: LLM inference server that runs on commodity hardware",
            "The reproducibility crisis in AI — better than ever, or worse?",
        ],
    },

    "Embedded Systems (C/RTOS)": {
        "languages": ["C", "C++", "Assembly", "Rust"],
        "subreddits": ["embedded", "RTOS", "arduino", "rust_embedded"],
        "stars_mu": 6.5, "stars_sigma": 2.1,
        "contrib_range": (2, 100), "gfi_prob": 0.15,
        "keywords": ["RTOS", "microcontroller", "bare-metal", "interrupt", "driver",
                     "HAL", "Raspberry Pi", "Arduino", "STM32", "FreeRTOS",
                     "memory-mapped", "DMA", "SPI", "I2C", "UART"],
        "repo_names": [
            ("freertos-project-template", "FreeRTOS project template for STM32 with task management, semaphores, and peripheral drivers"),
            ("embedded-rust-starter", "Embedded Rust starter kit for ARM Cortex-M with BSP, HAL drivers, and RTOS integration"),
            ("bare-metal-os-tutorial", "Build a bare-metal OS from scratch: bootloader, memory management, and task scheduling"),
            ("stm32-driver-library", "Comprehensive STM32 peripheral driver library: SPI, I2C, UART, DMA with interrupt handling"),
            ("rtos-scheduler-analysis", "RTOS scheduler analysis tool: task timing analysis, stack usage, and preemption visualization"),
            ("embedded-testing-framework", "Unit testing framework for embedded C with mock hardware, register simulation, and CI support"),
            ("can-bus-library", "CAN bus library for automotive applications with error handling, message filtering, and diagnostics"),
            ("low-power-firmware-guide", "Low-power firmware design guide: sleep modes, peripheral management, and power profiling tools"),
            ("bootloader-from-scratch", "Custom bootloader implementation for ARM Cortex-M: firmware update over UART, CRC verification"),
            ("real-time-audio-embedded", "Real-time audio processing on STM32: FFT, filters, and audio effects at 48kHz sample rate"),
            ("embedded-filesystem", "Embedded filesystem for microcontrollers: wear leveling, journaling, and FAT32 compatibility"),
            ("sensor-fusion-library", "Sensor fusion library: complementary filter, Kalman filter, and Madgwick algorithm for IMU"),
            ("embedded-networking-stack", "Lightweight networking stack for embedded systems: TCP/IP, MQTT, and TLS on microcontrollers"),
            ("debug-probe-firmware", "Open-source debug probe firmware using RP2040 as SWD/JTAG debugger for Cortex-M targets"),
            ("ota-update-framework", "Over-the-air firmware update framework with dual-bank flash, rollback support, and delta patches"),
        ],
        "reddit_posts": [
            "Best embedded systems projects for learning RTOS concepts",
            "How to get started contributing to embedded Linux or FreeRTOS",
            "Show r/embedded: Built a custom RTOS from scratch for Cortex-M4",
            "Resources for learning bare-metal programming in C in 2026",
            "Ask r/embedded: What's the best way to test embedded firmware?",
            "Monthly: Open-source embedded projects worth contributing to",
            "Rust for embedded systems — is it ready for production in 2026?",
            "The challenges of contributing to hardware-dependent open-source",
            "How I contributed to Zephyr RTOS — a beginner's guide",
            "Embedded debugging techniques that every developer should know",
        ],
        "hn_posts": [
            "Ask HN: What's the best resource for learning embedded systems programming?",
            "Show HN: RTOS written in Rust that passes DO-178C Level A certification tests",
            "Why embedded programming skills are more valuable than ever",
            "Show HN: Running ML inference on a $2 microcontroller",
            "The hidden complexity of embedded systems — a reflection",
        ],
    },

    "Cloud APIs": {
        "languages": ["Python", "TypeScript", "Go", "Java", "Terraform"],
        "subreddits": ["aws", "googlecloud", "azure", "cloudcomputing"],
        "stars_mu": 7.6, "stars_sigma": 2.0,
        "contrib_range": (5, 400), "gfi_prob": 0.25,
        "keywords": ["AWS", "Google Cloud", "Azure", "serverless", "Lambda",
                     "Cloud Run", "API Gateway", "IAM", "VPC", "S3",
                     "cloud-native", "cost optimization", "multi-cloud", "CDK"],
        "repo_names": [
            ("aws-cdk-patterns", "AWS CDK construct library with well-architected patterns for serverless, containers, and data"),
            ("cloud-cost-optimizer", "Multi-cloud cost optimization tool with unused resource detection and rightsizing recommendations"),
            ("terraform-cloud-modules", "Cloud-agnostic Terraform modules for GCP, AWS, and Azure with security and compliance controls"),
            ("serverless-framework-toolkit", "Serverless framework toolkit with local emulation, deployment automation, and observability"),
            ("cloud-run-templates", "Google Cloud Run deployment templates with CI/CD, secrets management, and global load balancing"),
            ("aws-lambda-powertools", "AWS Lambda powertools extensions: structured logging, tracing, and middleware for Python/TypeScript"),
            ("iam-policy-generator", "IAM policy generator using least-privilege principles with visual policy editor and simulation"),
            ("cloud-native-patterns", "Cloud-native application patterns: circuit breaker, bulkhead, retry with jitter, and chaos testing"),
            ("multi-cloud-storage-lib", "Unified storage API abstracting AWS S3, GCS, and Azure Blob with streaming and multipart support"),
            ("api-gateway-templates", "API Gateway configuration templates for REST, GraphQL, and gRPC with auth and rate limiting"),
            ("cloud-security-posture", "Cloud security posture management: drift detection, compliance checks, and auto-remediation"),
            ("kubernetes-cloud-controllers", "Cloud-specific Kubernetes controllers: ELB, Route53, ACM, and EFS integration"),
            ("cloud-observability-stack", "Cloud observability stack with OpenTelemetry, managed Prometheus, and Grafana dashboards"),
            ("event-driven-architecture", "Event-driven architecture reference using SNS/SQS, Pub/Sub, and Event Grid patterns"),
            ("cloud-migration-toolkit", "Cloud migration toolkit: assessment, dependency mapping, and automated infrastructure provisioning"),
        ],
        "reddit_posts": [
            "Best open-source tools for managing AWS costs in 2026",
            "Show r/aws: Built a CDK construct library with 2k downloads/week",
            "Contributing to cloud-related open-source — best projects to start with",
            "Terraform vs CDK vs Pulumi — what does the industry prefer in 2026?",
            "How to find security misconfigurations in your cloud infrastructure",
            "Ask r/googlecloud: Best resources for Cloud Run and GKE contributions?",
            "Monthly: Cloud tools and frameworks you should know about",
            "Multi-cloud strategy — is it worth the complexity?",
            "Serverless vs containers — where does the industry stand in 2026?",
            "Show r/cloudcomputing: Open-source cloud cost dashboard — feedback welcome",
        ],
        "hn_posts": [
            "Ask HN: What's the best way to reduce AWS costs without sacrificing reliability?",
            "Show HN: Open-source Terraform module library used by 500+ companies",
            "The cloud cost transparency problem — and how open-source helps",
            "Show HN: Serverless framework that cut our Lambda cold starts by 80%",
            "Is multi-cloud worth it? A field report from 3 years of practice",
        ],
    },

    "Mobile Dev (iOS/Flutter)": {
        "languages": ["Swift", "Dart", "Kotlin", "Objective-C"],
        "subreddits": ["iOSProgramming", "FlutterDev", "androiddev", "SwiftUI"],
        "stars_mu": 7.3, "stars_sigma": 2.1,
        "contrib_range": (3, 200), "gfi_prob": 0.28,
        "keywords": ["SwiftUI", "Flutter", "Dart", "iOS", "Android", "Jetpack Compose",
                     "mobile performance", "accessibility", "offline-first",
                     "push notifications", "deep linking", "widget"],
        "repo_names": [
            ("swiftui-component-library", "SwiftUI component library with accessibility support, dark mode, and dynamic type scaling"),
            ("flutter-state-management", "Flutter state management patterns: BLoC, Riverpod, and Redux with examples and benchmarks"),
            ("ios-architecture-patterns", "iOS architecture pattern examples: MVVM, VIPER, Clean Architecture with unit test examples"),
            ("flutter-animation-kit", "Flutter animation kit with physics-based animations, custom painters, and gesture handling"),
            ("mobile-offline-sync", "Mobile offline-first sync library for Flutter and iOS with conflict resolution and background sync"),
            ("swiftui-charts-library", "SwiftUI charts library with line, bar, pie, and candlestick charts with interactive gestures"),
            ("flutter-permission-handler", "Cross-platform permission handler for Flutter with explanation dialogs and settings navigation"),
            ("ios-networking-layer", "iOS networking layer with Combine/async-await, caching, retry logic, and certificate pinning"),
            ("flutter-local-notifications", "Flutter local notifications with scheduling, channels, and cross-platform payload handling"),
            ("mobile-performance-toolkit", "Mobile performance toolkit: memory profiling, frame rate analysis, and startup time measurement"),
            ("swiftui-navigation-patterns", "SwiftUI navigation patterns: deep links, coordinator pattern, and sheet/full-screen flows"),
            ("flutter-design-system", "Flutter design system generator from Figma tokens with automated theming and documentation"),
            ("ios-widget-extension", "iOS widget extension examples: timeline providers, configuration intents, and deeplink handling"),
            ("cross-platform-camera", "Cross-platform camera library for Flutter with ML integration, filters, and media management"),
            ("mobile-analytics-sdk", "Privacy-first mobile analytics SDK for iOS and Flutter with offline queuing and consent management"),
        ],
        "reddit_posts": [
            "Best SwiftUI open-source projects to contribute to in 2026",
            "Show r/FlutterDev: Built an animation library — looking for contributors",
            "Flutter vs SwiftUI vs Jetpack Compose — which should you learn in 2026?",
            "How to find beginner-friendly issues in iOS/Flutter open-source projects",
            "Ask r/iOSProgramming: What's the hardest part of contributing to iOS libraries?",
            "Monthly: Best mobile development resources and tools this month",
            "iOS offline-first architecture — best patterns and libraries",
            "Contributing to Flutter's core — a first-timer's experience",
            "SwiftUI performance tips that every developer should know",
            "Show r/FlutterDev: Released my first Flutter package — 200 installs!",
        ],
        "hn_posts": [
            "Ask HN: Flutter vs native in 2026 — what do you recommend?",
            "Show HN: SwiftUI component library with 100% accessibility coverage",
            "The state of cross-platform mobile development in 2026",
            "Show HN: Flutter package that makes offline-first apps trivial to build",
            "Why iOS development is getting easier — and what still hurts",
        ],
    },

    "Beginner Coding": {
        "languages": ["Python", "JavaScript", "Java", "C", "Ruby"],
        "subreddits": ["learnprogramming", "learnpython", "cs50", "beginnerprojects"],
        "stars_mu": 7.0, "stars_sigma": 2.2,
        "contrib_range": (3, 400), "gfi_prob": 0.65,
        "keywords": ["beginner", "tutorial", "learn to code", "good first issue",
                     "documentation", "example", "starter", "exercises",
                     "algorithms", "data structures", "practice", "challenge"],
        "repo_names": [
            ("coding-exercises-python", "500+ Python coding exercises from beginner to intermediate with tests and solutions"),
            ("algorithms-visualizer", "Interactive algorithm visualizer: sorting, pathfinding, and graph traversal with step-by-step explanation"),
            ("web-dev-roadmap-projects", "Project-based web development learning: 30 projects from HTML basics to full-stack applications"),
            ("data-structures-explained", "Data structures with clear explanations, diagrams, and implementations in Python, Java, and C++"),
            ("open-source-for-beginners", "Curated list of beginner-friendly open-source projects across languages and domains"),
            ("first-contributions-guide", "Step-by-step guide to making your first open-source contribution with a practice repository"),
            ("leetcode-patterns", "LeetCode problem patterns for interview preparation with Python and JavaScript solutions"),
            ("build-your-own-x", "Build your own programming tools: shell, database, git, HTTP server — with detailed tutorials"),
            ("python-mini-projects", "100 beginner Python mini-projects with increasing complexity, tests, and video walkthroughs"),
            ("javascript-30-projects", "30 JavaScript projects in 30 days: DOM manipulation, APIs, games, and utilities"),
            ("cs-fundamentals-repo", "Computer science fundamentals: algorithms, complexity analysis, and system design for beginners"),
            ("coding-interview-handbook", "Coding interview handbook with patterns, tips, and practice problems for new developers"),
            ("github-profile-templates", "Beautiful GitHub profile README templates with contribution graphs, stats, and badges"),
            ("documentation-examples", "Real-world documentation examples across languages — learn to write docs that people read"),
            ("testing-for-beginners", "Testing concepts for beginners: unit tests, TDD, and test-driven design with Python examples"),
        ],
        "reddit_posts": [
            "What was your first open-source contribution and how did you find it?",
            "Ask r/learnprogramming: How do I find issues suitable for my skill level?",
            "I made my first open-source PR — here's what I learned",
            "Best repositories for beginner programmers to explore and contribute",
            "Show r/learnprogramming: My portfolio project — feedback welcome",
            "How to go from tutorial hell to actually building things",
            "Monthly: Beginner-friendly repositories that need contributors",
            "Resources for learning how to read and navigate large codebases",
            "I went from zero to first PR in 30 days — here's my journey",
            "Ask r/learnpython: What's the best open-source Python project for beginners?",
        ],
        "hn_posts": [
            "Ask HN: What resources helped you most as a beginner programmer?",
            "Show HN: I built a platform for first-time open-source contributors",
            "Why beginner-friendly issues are hard to write and how to improve",
            "Show HN: Visual algorithm explorer built by a CS student",
            "How open-source contributions accelerated my career as a junior developer",
        ],
    },
}

DOMAINS = list(DOMAIN_CFG.keys())


def make_id(source: str, domain: str, idx: int) -> str:
    raw = f"{source}:{domain}:{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def random_date(days_back: int = 365) -> str:
    delta = timedelta(days=random.randint(0, days_back))
    return (datetime.now() - delta).strftime("%Y-%m-%d %H:%M:%S")


def compute_activity_score(stars, forks, contributors, comments, growth_rate) -> float:
    s = min(stars / 10000, 1.0) * 0.25
    f = min(forks / 2000, 1.0) * 0.15
    c = min(contributors / 200, 1.0) * 0.20
    co = min(comments / 500, 1.0) * 0.15
    g = min(max(growth_rate, 0) / 100, 1.0) * 0.25
    return round(s + f + c + co + g, 4)


def generate_github_records(domain: str, cfg: dict, n: int = 400) -> list[dict]:
    records = []
    repo_pool = cfg["repo_names"]
    keywords = cfg["keywords"]
    languages = cfg["languages"]

    for i in range(n):
        template_idx = i % len(repo_pool)
        base_name, base_desc = repo_pool[template_idx]
        suffix = f"-v{i // len(repo_pool) + 1}" if i >= len(repo_pool) else ""
        name = base_name + suffix

        kw = random.choice(keywords)
        desc = base_desc + f" — includes {kw} support and comprehensive documentation."

        stars = max(10, int(np.random.lognormal(cfg["stars_mu"], cfg["stars_sigma"])))
        forks = max(0, int(stars * random.uniform(0.05, 0.25)))
        contributors = random.randint(*cfg["contrib_range"])
        open_issues = max(0, int(stars * random.uniform(0.01, 0.08)))
        gfi = random.randint(1, 15) if random.random() < cfg["gfi_prob"] else 0
        growth = round(random.uniform(-5, stars * 0.05), 2)
        lang = random.choice(languages)
        tags = json.dumps(random.sample(keywords, min(4, len(keywords))))

        record = {
            "id": make_id("github", domain, i + 1000 * DOMAINS.index(domain)),
            "source": "github",
            "record_type": "issue" if gfi > 0 and random.random() < 0.3 else "repo",
            "data_source": "offline",
            "title": f"{name}",
            "description": desc,
            "url": f"https://github.com/search?q={quote(name.replace(' ', '-'))}&type=repositories",
            "domain": domain,
            "language": lang,
            "tags": tags,
            "stars": stars,
            "forks": forks,
            "contributors": contributors,
            "open_issues": open_issues,
            "good_first_issues": gfi,
            "comments": open_issues,
            "upvotes": 0,
            "activity_score": compute_activity_score(stars, forks, contributors, open_issues, growth),
            "growth_rate": growth,
            "created_at": random_date(730),
            "updated_at": random_date(30),
        }
        records.append(record)
    return records


def generate_hn_records(domain: str, cfg: dict, n: int = 100) -> list[dict]:
    records = []
    posts = cfg["hn_posts"]
    keywords = cfg["keywords"]

    for i in range(n):
        template_idx = i % len(posts)
        base_title = posts[template_idx]
        variation = i // len(posts)
        if variation > 0:
            base_title = base_title + f" ({variation + 1})"

        score = max(1, int(np.random.lognormal(4.5, 1.5)))
        comments = max(0, int(score * random.uniform(0.3, 1.5)))
        kw = random.choice(keywords)
        tags = json.dumps(random.sample(keywords, min(3, len(keywords))))

        record = {
            "id": make_id("hackernews", domain, i + 3000 * DOMAINS.index(domain)),
            "source": "hackernews",
            "record_type": "hn_story",
            "data_source": "offline",
            "title": base_title,
            "description": f"Hacker News discussion: {base_title} — score: {score}, {comments} comments. Covers {kw} and related {domain} topics.",
            "url": f"https://hn.algolia.com/?query={quote(base_title[:80])}&type=story",
            "domain": domain,
            "language": "",
            "tags": tags,
            "stars": 0,
            "forks": 0,
            "contributors": 0,
            "open_issues": 0,
            "good_first_issues": 0,
            "comments": comments,
            "upvotes": score,
            "activity_score": compute_activity_score(0, 0, 0, comments, score / 50),
            "growth_rate": round(random.uniform(0, 30), 2),
            "created_at": random_date(60),
            "updated_at": random_date(3),
        }
        records.append(record)
    return records


def main():
    all_records = []

    print("Generating offline dataset (GitHub + Hacker News)...")
    for domain in DOMAINS:
        cfg = DOMAIN_CFG[domain]
        gh = generate_github_records(domain, cfg, n=500)
        hn = generate_hn_records(domain, cfg, n=200)
        all_records.extend(gh + hn)
        print(f"  {domain}: {len(gh)} GitHub + {len(hn)} HN = {len(gh)+len(hn)}")

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["id"])
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(df)} records to {OUT_CSV}")
    print(f"Domains covered: {df['domain'].nunique()}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    print(f"data_source: {df['data_source'].value_counts().to_dict()}")
    print("\nRecord counts by domain:")
    for domain, count in df.groupby("domain").size().items():
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
