"""A small synthetic labelled corpus for model tests. All text is invented."""

from __future__ import annotations

import random

from sensclf.labels import CATEGORIES
from sensclf.splits import LabelledDocument

# Each category has a cue phrase; a document contains the cues of its positive categories.
CUES = {
    "IDENTITY": "full name and date of birth",
    "CONTACT": "email address and phone number",
    "FINANCIAL": "credit card and bank account",
    "HEALTH": "medical record and blood type",
    "CREDENTIALS": "password and api key",
    "SPECIAL_CATEGORY": "religious belief and political view",
}
FILLER = ["the quarterly report", "meeting notes follow", "please review the attached",
          "thanks for your help", "see you on monday"]
DOMAINS = ["Healthcare", "Finance", "Technology"]


def toy_docs(n: int = 60, seed: int = 0, prefix: str = "toy") -> list[LabelledDocument]:
    """``n`` documents; every category is positive in some and negative in others."""
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        labels = {c: int(rng.random() < 0.4) for c in CATEGORIES}
        labels[CATEGORIES[i % len(CATEGORIES)]] = 1          # every category has positives
        parts = [rng.choice(FILLER)] + [CUES[c] for c in CATEGORIES if labels[c]]
        rng.shuffle(parts)
        domain = DOMAINS[i % len(DOMAINS)]
        docs.append(LabelledDocument(f"{prefix}:{i}", ". ".join(parts) + ".", labels,
                                     {"domain": domain, "group_id": f"{prefix}-g{i // 2}"}))
    return docs
