"""
Adaptive learning via Thompson Sampling bandit.
BAX-423 Lecture 8 — Introduction to Reinforcement Learning (Exploration vs Exploitation).

Thompson Sampling maintains a Beta distribution per opportunity.
Feedback signals: engage (+1), bookmark (+0.5), skip (−0.5 via beta increment).
Domain preference learning tracks which domains the user engages with most.
"""
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).parent
STATE_FILE = _HERE.parent / "data" / "bandit_state.json"


@dataclass
class ThompsonBandit:
    """
    Beta-Bernoulli bandit using Thompson Sampling.
    alpha[id] = success count (engage/bookmark)
    beta[id] = failure count (skip)
    """
    alpha: dict = field(default_factory=dict)
    beta_: dict = field(default_factory=dict)
    domain_engages: dict = field(default_factory=lambda: defaultdict(float))
    domain_skips: dict = field(default_factory=lambda: defaultdict(float))
    total_rounds: int = 0

    def _get_alpha(self, opp_id: str) -> float:
        return self.alpha.get(opp_id, 1.0)

    def _get_beta(self, opp_id: str) -> float:
        return self.beta_.get(opp_id, 1.0)

    def sample_score(self, opp_id: str) -> float:
        """Sample from posterior Beta(alpha, beta) for this opportunity."""
        a = self._get_alpha(opp_id)
        b = self._get_beta(opp_id)
        return float(np.random.beta(a, b))

    def get_bandit_scores(self, opp_ids: list[str]) -> dict[str, float]:
        """Return posterior means for ranking — no randomness for display."""
        scores = {}
        for opp_id in opp_ids:
            a = self._get_alpha(opp_id)
            b = self._get_beta(opp_id)
            scores[opp_id] = a / (a + b)
        return scores

    def update(self, opp_id: str, feedback: str, domain: str = ""):
        """Update posteriors based on user feedback."""
        self.total_rounds += 1
        if feedback == "engage":
            self.alpha[opp_id] = self._get_alpha(opp_id) + 1.0
            if domain:
                self.domain_engages[domain] += 1.0
        elif feedback == "bookmark":
            self.alpha[opp_id] = self._get_alpha(opp_id) + 0.5
            if domain:
                self.domain_engages[domain] += 0.5
        elif feedback == "skip":
            self.beta_[opp_id] = self._get_beta(opp_id) + 1.0
            if domain:
                self.domain_skips[domain] += 0.5

    def get_domain_preferences(self) -> dict[str, float]:
        """
        Normalized domain preference scores in [0, 1].
        Used as a ranking boost in the multi-stage pipeline.
        """
        all_domains = set(list(self.domain_engages.keys()) + list(self.domain_skips.keys()))
        if not all_domains:
            return {}
        prefs = {}
        for d in all_domains:
            e = self.domain_engages.get(d, 0)
            s = self.domain_skips.get(d, 0)
            total = e + s
            prefs[d] = e / total if total > 0 else 0.5
        # Normalize to [0, 1]
        vals = list(prefs.values())
        if max(vals) > min(vals):
            prefs = {d: (v - min(vals)) / (max(vals) - min(vals)) for d, v in prefs.items()}
        return prefs

    def simulate_feedback_rounds(
        self, opportunity_df, n_rounds: int = 50, persona_domains: list = None
    ) -> list[dict]:
        """
        Simulate n feedback rounds for a synthetic persona.
        Returns per-round metrics to demonstrate learning improvement.
        """
        metrics = []
        persona_domains = set(persona_domains or [])
        sample_ids = opportunity_df["id"].tolist()
        id_to_domain = dict(zip(opportunity_df["id"], opportunity_df["domain"]))

        for round_idx in range(n_rounds):
            # Sample 10 opportunities using Thompson Sampling
            candidates = random.sample(sample_ids, min(50, len(sample_ids)))
            ts_scores = [(oid, self.sample_score(oid)) for oid in candidates]
            ts_scores.sort(key=lambda x: x[1], reverse=True)
            top_10 = ts_scores[:10]

            # Synthetic persona gives feedback
            for opp_id, _ in top_10:
                domain = id_to_domain.get(opp_id, "")
                if domain in persona_domains:
                    feedback = "engage" if random.random() < 0.7 else "bookmark"
                else:
                    feedback = "skip"
                self.update(opp_id, feedback, domain)

            # Precision@10: fraction of top-10 in persona's preferred domains
            p_at_10 = sum(1 for oid, _ in top_10 if id_to_domain.get(oid, "") in persona_domains) / 10
            metrics.append({"round": round_idx + 1, "precision_at_10": p_at_10})

        return metrics

    def precision_at_k(self, ranked_ids: list[str], relevant_domains: set, k: int = 10) -> float:
        hits = sum(1 for oid in ranked_ids[:k])
        return hits / k if k > 0 else 0.0

    def save(self, path: Path = STATE_FILE):
        path.parent.mkdir(exist_ok=True)
        state = {
            "alpha": self.alpha,
            "beta_": self.beta_,
            "domain_engages": dict(self.domain_engages),
            "domain_skips": dict(self.domain_skips),
            "total_rounds": self.total_rounds,
        }
        path.write_text(json.dumps(state))

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> "ThompsonBandit":
        if path.exists():
            try:
                state = json.loads(path.read_text())
                b = cls()
                b.alpha = state.get("alpha", {})
                b.beta_ = state.get("beta_", {})
                b.domain_engages = defaultdict(float, state.get("domain_engages", {}))
                b.domain_skips = defaultdict(float, state.get("domain_skips", {}))
                b.total_rounds = state.get("total_rounds", 0)
                return b
            except Exception:
                pass
        return cls()
