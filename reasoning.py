"""LLM Reasoning Layer for the AI Due Diligence Agent.

Design principle: the LLM does not create numbers or judge risk from raw text.
It only narrates and contextualizes the deterministic flags and extracted facts
produced by the Python engine. It must cite its sources, and every citation is
validated against the data actually provided. If the audit fails, the output is
rejected and regenerated with a correction — never shown unsourced.
"""

import re
import time

SYSTEM_PROMPT = (
    "You are an elite Private Equity due diligence analyst. "
    "Your task is to write a concise, rigorous Red Flag Report based ONLY "
    "on the structured data provided to you.\n\n"
    "CRITICAL RULES:\n"
    "1. You may ONLY use the facts, metrics, and flags provided.\n"
    "2. Do not invent numbers. Do not guess risks.\n"
    "3. Every time you mention a number or a risk flag, you MUST cite it "
    "using the provided ID tags in brackets, like this: [Fact-1] or [Flag-2] "
    "or [Metric-3].\n"
    "4. If you cannot cite a claim, do not write it.\n"
    "5. Write in a professional, institutional tone. Use markdown."
)

QA_SYSTEM_PROMPT = (
    "You are an elite Private Equity due diligence analyst. "
    "Answer the user's question using ONLY the structured data store provided "
    "(document metadata, extracted facts, derived metrics, risk flags). "
    "Do not use outside knowledge. Do not read raw documents.\n\n"
    "CRITICAL RULES:\n"
    "1. If the data needed to answer is missing, state exactly what is missing "
    "and do not guess.\n"
    "2. Cite every number and flag with [Fact-X], [Metric-X], or [Flag-X] tags.\n"
    "3. Metadata answers (company name, file name, document type) do not need "
    "tags.\n"
    "4. Be concise, rigorous, and professional. Use markdown."
)


def build_context(facts, metrics, flags, metadata=None):
    """Format the deterministic data into a strict context block for the LLM."""
    context = ""

    if metadata:
        context += "### DOCUMENT METADATA\n"
        for key, value in metadata.items():
            if value not in (None, ""):
                context += f"{key}: {value}\n"
        context += "\n"

    context += "### EXTRACTED FACTS\n"

    fact_ids = {}
    for i, fact in enumerate(facts, 1):
        tag = f"Fact-{i}"
        fact_ids[tag] = fact
        val = fact.get("value")
        val_str = "NULL" if val is None else f"{val} {fact.get('unit', '')}"
        context += (
            f"[{tag}] {fact.get('period')} {fact.get('statement')}."
            f"{fact.get('field')}: {val_str} (Page {fact.get('source_page')})\n"
        )

    context += "\n### DERIVED METRICS (Calculated in Python)\n"

    metric_ids = {}
    for i, metric in enumerate(metrics, 1):
        tag = f"Metric-{i}"
        metric_ids[tag] = metric
        context += (
            f"[{tag}] {metric.get('period')} {metric.get('metric_name')}: "
            f"{metric.get('value')}\n"
        )

    context += "\n### DETERMINISTIC RISK FLAGS (Calculated in Python)\n"

    flag_ids = {}
    for i, flag in enumerate(flags, 1):
        tag = f"Flag-{i}"
        flag_ids[tag] = flag
        context += (
            f"[{tag}] [{flag.get('severity').upper()}] {flag.get('rule')}: "
            f"{flag.get('explanation')}\n"
        )

    return context, fact_ids, metric_ids, flag_ids


def _generate_once(client, model, prompt_text):
    """One generation call with retry on temporary free-tier overload."""
    from google.genai import types

    last_error = None

    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": [{"text": prompt_text}]}],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                ),
            )
        except Exception as error:
            last_error = error
            text = str(error)

            retryable = (
                "503" in text
                or "429" in text
                or "UNAVAILABLE" in text
                or "RESOURCE_EXHAUSTED" in text
            )

            if retryable and attempt < 2:
                time.sleep(20)
                continue

            raise

    raise last_error


def _audit_tags(raw_text, all_valid_tags):
    used_tags = {
        f"[{kind}-{number}]"
        for kind, number in re.findall(r"\[(Fact|Metric|Flag)-(\d+)\]", raw_text)
    }
    return used_tags - all_valid_tags


def _generate_with_audit(client, model, base_prompt, all_valid_tags, max_audit_retries=2):
    """Generate, audit citations, and regenerate with a correction on failure."""
    correction = ""
    hallucinated = set()
    raw_text = ""

    for attempt in range(max_audit_retries + 1):
        response = _generate_once(client, model, base_prompt + correction)
        raw_text = response.text

        hallucinated = _audit_tags(raw_text, all_valid_tags)

        if not hallucinated:
            return raw_text, set()

        correction = (
            "\n\nCORRECTION FROM GOVERNANCE LAYER: your previous draft used "
            "citation tags that do not exist in the data: "
            + ", ".join(sorted(hallucinated))
            + ". You may ONLY use these tags: "
            + ", ".join(sorted(all_valid_tags))
            + ". Regenerate the complete answer now, citing only valid tags."
        )

    return raw_text, hallucinated


def generate_red_flag_report(facts, metrics, flags, api_key, model, metadata=None):
    """Narrate the deterministic outputs as an institutional red-flag report."""
    from google import genai

    client = genai.Client(api_key=api_key)

    context, fact_ids, metric_ids, flag_ids = build_context(
        facts, metrics, flags, metadata
    )

    user_prompt = (
        "Here is the structured data for the target company:\n\n"
        f"{context}\n\n"
        "Write a 3-section Red Flag Report:\n"
        "1. Executive Summary (2-3 sentences, name the company if metadata "
        "provides it)\n"
        "2. Key Financial Risks (Discuss the active flags, citing the evidence)\n"
        "3. Missing Data / Blind Spots (Mention the dormant rules and why we "
        "couldn't assess them)\n\n"
        "Remember: Cite every number and flag using the [Fact-X], [Metric-X], "
        "or [Flag-X] tags."
    )

    all_valid_tags = {
        f"[{tag}]"
        for tag in (set(fact_ids.keys()) | set(metric_ids.keys()) | set(flag_ids.keys()))
    }

    return _generate_with_audit(
        client, model, SYSTEM_PROMPT + "\n\n" + user_prompt, all_valid_tags
    )


def answer_question(question, facts, metrics, flags, api_key, model, metadata=None):
    """Answer a user question strictly from the structured data store."""
    from google import genai

    client = genai.Client(api_key=api_key)

    context, fact_ids, metric_ids, flag_ids = build_context(
        facts, metrics, flags, metadata
    )

    base_prompt = (
        QA_SYSTEM_PROMPT
        + "\n\n### STRUCTURED DATA STORE\n"
        + context
        + "\n\nQUESTION: "
        + question
    )

    all_valid_tags = {
        f"[{tag}]"
        for tag in (set(fact_ids.keys()) | set(metric_ids.keys()) | set(flag_ids.keys()))
    }

    return _generate_with_audit(client, model, base_prompt, all_valid_tags)