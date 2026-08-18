"""Document classification for the AI Due Diligence Agent.

Deterministic keyword scoring over the FULL document first. If the keyword
evidence is strong, the deterministic result is final (fast, auditable,
free). If the evidence is weak or ambiguous, an LLM confirms using an
informative page sample (not just the front matter). The method used is
always recorded honestly.
"""

import json

DOC_TYPES = [
    "financial_statement",
    "contract",
    "presentation",
    "transcript",
    "other",
]

KEYWORDS = {
    "transcript": [
        "operator:",
        "earnings call",
        "conference call",
        "analyst",
        "good morning",
        "question-and-answer",
        "management call",
    ],
    "contract": [
        "agreement",
        "hereinafter",
        "whereas",
        "termination",
        "indemnification",
        "arbitration",
        "clause",
        "parties",
    ],
    "financial_statement": [
        "balance sheet",
        "income statement",
        "statement of profit",
        "cash flow",
        "statement of financial position",
        "audited financial",
        "notes to accounts",
    ],
    "presentation": [
        "slide",
        "agenda",
        "investor presentation",
        "highlights",
        "overview",
    ],
}

STRONG_MIN_COUNT = 5
STRONG_MIN_SHARE = 0.4

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string", "enum": DOC_TYPES},
        "confidence": {"type": "number"},
    },
    "required": ["doc_type", "confidence"],
}


def deterministic_classify(pages):
    """Score the FULL document text with keyword patterns."""
    full_text = "\n".join((text or "") for _, text in pages).lower()

    scores = {
        doc_type: sum(full_text.count(keyword) for keyword in keywords)
        for doc_type, keywords in KEYWORDS.items()
    }

    best_type = max(scores, key=scores.get)
    total = sum(scores.values())

    if total == 0:
        return "other", 0.0, scores

    return best_type, round(scores[best_type] / total, 4), scores


def _informative_sample(pages, max_pages=6):
    """Pick the most keyword-dense pages, so the LLM never judges a 300-page
    filing by its cover pages."""
    scored = []

    for page_number, text in pages:
        lowered = (text or "").lower()
        score = sum(
            lowered.count(keyword)
            for keywords in KEYWORDS.values()
            for keyword in keywords
        )
        scored.append((score, page_number, text))

    scored.sort(key=lambda item: -item[0])
    top = sorted(scored[:max_pages], key=lambda item: item[1])

    return "\n".join((text or "")[:3000] for _, _, text in top)


def classify_document(pages, api_key=None, model=None):
    """Classify a document. Returns (doc_type, confidence, method)."""
    det_type, det_share, scores = deterministic_classify(pages)

    strong = (
        scores[det_type] >= STRONG_MIN_COUNT
        and det_share >= STRONG_MIN_SHARE
    )

    if strong or not api_key or not model:
        return det_type, det_share, "deterministic"

    from google import genai
    from google.genai import types

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents=(
                "Classify this corporate document into exactly one of these "
                f"types: {DOC_TYPES}.\n"
                "Important: a long annual report whose front matter reads like "
                "a presentation must still be classified as financial_statement "
                "if it contains audited financial statements. Judge the whole "
                "document, not just the first pages.\n"
                "Return the type and a confidence between 0 and 1.\n\n"
                + _informative_sample(pages)
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CLASSIFICATION_SCHEMA,
            ),
        )

        parsed = json.loads(response.text)

        doc_type = parsed.get("doc_type")
        confidence = parsed.get("confidence")

        if doc_type not in DOC_TYPES:
            return det_type, det_share, "deterministic"

        return doc_type, confidence, "llm"
    except Exception:
        return det_type, det_share, "deterministic"