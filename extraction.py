"""Structured financial extraction for the AI Due Diligence Agent.

Design principle: the LLM only extracts values that are explicitly stated
in the document. It never estimates or calculates. All ratios and metrics
are computed deterministically in Python, never by the LLM.

LLM provider: Google Gemini (free tier), with automatic retry on overload.
"""

import json
import time

PAGE_MARKER = "--- PAGE {page} ---"

FINANCIAL_FIELDS = {
    "income_statement": [
        "revenue",
        "cost_of_revenue",
        "gross_profit",
        "operating_income",
        "ebitda",
        "depreciation_and_amortization",
        "interest_expense",
        "tax_expense",
        "net_income",
        "cost_of_materials",
        "employee_benefit_expense",
        "other_expenses",
    ],
    "balance_sheet": [
        "cash_and_equivalents",
        "trade_receivables",
        "inventory",
        "property_plant_equipment",
        "total_current_assets",
        "total_current_liabilities",
        "short_term_debt",
        "long_term_debt",
        "total_debt",
        "total_equity",
    ],
    "cash_flow": [
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
    ],
    "disclosures": [
        "top_customer_revenue_pct",
    ],
}

ALL_FIELDS = [field for fields in FINANCIAL_FIELDS.values() for field in fields]

CANDIDATE_KEYWORDS = [
    "income statement",
    "balance sheet",
    "cash flow",
    "statement of profit",
    "statement of income",
    "statement of financial position",
    "ebitda",
    "revenue",
    "net income",
    "net profit",
    "profit for the year",
    "total comprehensive income",
    "earnings per share",
    "total debt",
    "operating activities",
    "profit before tax",
    "finance costs",
    "total income",
    "cash and cash equivalents",
    "total equity",
    "borrowings",
    "trade receivables",
    "inventories",
    "property, plant and equipment",
    "customer",
    "crore",
]

MAX_CANDIDATE_PAGES = 30
MAX_CHARS_PER_PAGE = 8000

SYSTEM_PROMPT = (
    "You are a precise financial statement extraction engine. "
    "You extract ONLY values that are explicitly stated in the provided "
    "document text. Never estimate, infer, or calculate a value. "
    "If a value is not explicitly stated, return null for that value. "
    "For every annual period you identify, include an entry for every "
    "requested field, using null for values that are not explicitly stated. "
    "Normalize amounts to millions in the document's reporting currency. "
    "Indian documents report in crore or lakh: 1 crore = 10 million, "
    "1 lakh = 0.1 million. Convert stated crore or lakh figures to millions. "
    "Map Ind AS and US GAAP line items as follows: "
    "'Revenue from operations' or 'Total income' -> revenue; "
    "'Finance costs' -> interest_expense; "
    "'Tax expense' or 'Income tax expense' -> tax_expense; "
    "'Profit for the period' or 'Net income' -> net_income; "
    "'Cash and cash equivalents' -> cash_and_equivalents; "
    "'Total equity' or 'Total shareholders equity' -> total_equity; "
    "'Trade receivables' or 'Receivables' -> trade_receivables; "
    "'Inventories' or 'Inventory' -> inventory; "
    "'Property, plant and equipment' -> property_plant_equipment; "
    "'Cost of materials' -> cost_of_materials; "
    "'Employee benefits expense' -> employee_benefit_expense; "
    "'Other expenses' -> other_expenses. "
    "For disclosures.top_customer_revenue_pct: if the document explicitly "
    "states a single customer's share of revenue (for example 'one customer "
    "accounted for 12% of revenue'), record that percentage as a number. "
    "If no such percentage is explicitly stated, return null. "
    "The extracted page text may contain imperfectly formatted tables. "
    "Use line labels, column headers and year headings to attach each number "
    "to the correct period. If the mapping is ambiguous, return null. "
    "For every extracted value, provide the page number shown in the "
    "PAGE markers and a short verbatim snippet from that page. "
    "Extract up to the three most recent annual periods available. "
    "Handle both US GAAP and Ind AS style disclosures."
)

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "statement": {"type": "string", "enum": list(FINANCIAL_FIELDS.keys())},
        "field": {"type": "string", "enum": ALL_FIELDS},
        "period": {"type": "string"},
        "value": {"type": "number", "nullable": True},
        "page": {"type": "integer", "nullable": True},
        "snippet": {"type": "string", "nullable": True},
        "confidence": {"type": "number", "nullable": True},
    },
    "required": [
        "statement",
        "field",
        "period",
        "value",
        "page",
        "snippet",
        "confidence",
    ],
}

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "currency": {"type": "string", "nullable": True},
        "unit": {"type": "string", "nullable": True},
        "facts": {"type": "array", "items": FACT_SCHEMA},
    },
    "required": ["currency", "unit", "facts"],
}


def find_candidate_pages(pages):
    """Deterministically rank pages likely to contain financial statements.

    pages: list of (page_number, text) tuples.
    Returns the top candidate pages in page order.
    """
    scored = []

    for page_number, text in pages:
        text_lower = (text or "").lower()
        score = sum(text_lower.count(keyword) for keyword in CANDIDATE_KEYWORDS)
        if score > 0:
            scored.append((score, page_number, text))

    scored.sort(key=lambda item: (-item[0], item[1]))
    top = sorted(scored[:MAX_CANDIDATE_PAGES], key=lambda item: item[1])

    return [(page_number, text) for _, page_number, text in top]


def build_extraction_input(candidate_pages):
    chunks = []

    for page_number, text in candidate_pages:
        chunks.append(PAGE_MARKER.format(page=page_number))
        chunks.append(text[:MAX_CHARS_PER_PAGE])

    return "\n".join(chunks)


def run_extraction(candidate_pages, api_key, model, max_retries=3):
    """Call Gemini to extract explicitly stated financial facts only.

    Automatically waits 20 seconds and retries up to 3 times when the
    free tier is overloaded (429/503).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    prompt = (
        SYSTEM_PROMPT
        + "\n\nExtract the following fields for each available annual period. "
        "Allowed fields per statement:\n"
        + json.dumps(FINANCIAL_FIELDS, indent=2)
        + "\n\nDocument text:\n"
        + build_extraction_input(candidate_pages)
    )

    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EXTRACTION_SCHEMA,
                    max_output_tokens=16384,
                ),
            )

            return json.loads(response.text)
        except Exception as error:
            last_error = error
            text = str(error)

            retryable = (
                "503" in text
                or "429" in text
                or "UNAVAILABLE" in text
                or "RESOURCE_EXHAUSTED" in text
            )

            if retryable and attempt < max_retries - 1:
                time.sleep(20)
                continue

            raise

    raise last_error


def compute_derived_metrics(facts):
    """Deterministic ratio calculation from extracted facts only.

    No LLM involvement. Same input always produces same output.
    """
    lookup = {}

    for fact in facts:
        if fact.get("value") is not None:
            lookup[(fact["period"], fact["field"])] = fact["value"]

    periods = sorted({fact["period"] for fact in facts}, reverse=True)
    metrics = []

    for period in periods:
        revenue = lookup.get((period, "revenue"))
        gross_profit = lookup.get((period, "gross_profit"))
        ebitda = lookup.get((period, "ebitda"))
        net_income = lookup.get((period, "net_income"))
        total_debt = lookup.get((period, "total_debt"))
        interest_expense = lookup.get((period, "interest_expense"))
        current_assets = lookup.get((period, "total_current_assets"))
        current_liabilities = lookup.get((period, "total_current_liabilities"))

        def ratio(numerator, denominator):
            if numerator is None or denominator in (None, 0):
                return None
            return round(numerator / denominator, 4)

        metrics.append(
            {
                "period": period,
                "gross_margin": ratio(gross_profit, revenue),
                "ebitda_margin": ratio(ebitda, revenue),
                "net_margin": ratio(net_income, revenue),
                "debt_to_ebitda": ratio(total_debt, ebitda),
                "interest_coverage": ratio(ebitda, interest_expense),
                "current_ratio": ratio(current_assets, current_liabilities),
            }
        )

    return metrics