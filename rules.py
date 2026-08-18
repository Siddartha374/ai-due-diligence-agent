"""Deterministic risk-rule engine for the AI Due Diligence Agent.

Pure Python. No LLM involvement. Same input always produces same output.
Every flag carries evidence pointing back to extracted facts (period, field,
value, source page), preserving the citation chain from extraction.
"""

from extraction import compute_derived_metrics

DEFAULT_THRESHOLDS = {
    "debt_to_ebitda_max": 4.0,
    "margin_compression_min_drop": 0.03,
    "current_ratio_min": 1.0,
    "customer_concentration_max_pct": 10.0,
}

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

DORMANT_RULES = [
    ("guidance_credibility", "Requires earnings transcript extraction (v2, Milestone 1J)."),
    ("related_party_transactions", "Requires contract and related-party note extraction (v2, Milestone 1K)."),
]


def _build_lookups(facts):
    values = {}
    pages = {}

    for fact in facts:
        if fact.get("value") is None:
            continue
        key = (fact["period"], fact["field"])
        values[key] = fact["value"]
        pages[key] = fact.get("source_page")

    metrics = {}
    for row in compute_derived_metrics(facts):
        period = row["period"]
        for metric_name, metric_value in row.items():
            if metric_name == "period" or metric_value is None:
                continue
            metrics[(period, metric_name)] = metric_value

    periods = sorted({period for period, _ in values.keys()})
    return values, pages, metrics, periods


def _evidence(values, pages, keys):
    return [
        {
            "period": period,
            "field": field,
            "value": values.get((period, field)),
            "page": pages.get((period, field)),
        }
        for period, field in keys
    ]


def rule_revenue_decline(values, pages, metrics, periods, thresholds):
    points = [(p, values.get((p, "revenue"))) for p in periods]
    points = [(p, v) for p, v in points if v is not None]

    if len(points) < 2:
        return []

    consecutive = 0
    max_consecutive = 0

    for (prev_p, prev_v), (next_p, next_v) in zip(points, points[1:]):
        if next_v < prev_v:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    if max_consecutive >= 2:
        severity = "high"
        explanation = f"Revenue declined for {max_consecutive + 1} consecutive periods."
    elif max_consecutive == 1:
        severity = "low"
        explanation = "Revenue declined in one period versus the prior period."
    else:
        return []

    return [
        {
            "rule": "revenue_decline",
            "severity": severity,
            "explanation": explanation,
            "evidence": _evidence(values, pages, [(p, "revenue") for p, _ in points]),
        }
    ]


def rule_margin_compression(values, pages, metrics, periods, thresholds):
    drop_min = thresholds["margin_compression_min_drop"]
    flags = []

    for prev, curr in zip(periods, periods[1:]):
        for metric in ("ebitda_margin", "net_margin"):
            m_prev = metrics.get((prev, metric))
            m_curr = metrics.get((curr, metric))

            if m_prev is None or m_curr is None:
                continue

            drop = m_prev - m_curr

            if drop >= drop_min:
                severity = "high" if drop >= 2 * drop_min else "medium"

                if metric == "ebitda_margin":
                    base_fields = ["revenue", "ebitda"]
                else:
                    base_fields = ["revenue", "net_income"]

                flags.append(
                    {
                        "rule": "margin_compression",
                        "severity": severity,
                        "explanation": (
                            f"{metric} compressed by {round(drop * 10000)} bps "
                            f"from {prev} to {curr} "
                            f"({round(m_prev * 100, 2)}% -> {round(m_curr * 100, 2)}%)."
                        ),
                        "evidence": _evidence(
                            values,
                            pages,
                            [
                                (prev, base_fields[0]),
                                (prev, base_fields[1]),
                                (curr, base_fields[0]),
                                (curr, base_fields[1]),
                            ],
                        ),
                    }
                )
                break

    return flags


def rule_leverage(values, pages, metrics, periods, thresholds):
    max_ratio = thresholds["debt_to_ebitda_max"]
    flags = []

    for period in periods:
        ratio = metrics.get((period, "debt_to_ebitda"))

        if ratio is None:
            continue

        if ratio > max_ratio:
            severity = "high" if ratio >= 1.5 * max_ratio else "medium"
            flags.append(
                {
                    "rule": "leverage",
                    "severity": severity,
                    "explanation": (
                        f"Debt/EBITDA of {round(ratio, 2)}x in {period} "
                        f"exceeds {max_ratio}x threshold."
                    ),
                    "evidence": _evidence(
                        values, pages, [(period, "total_debt"), (period, "ebitda")]
                    ),
                }
            )

    return flags


def rule_liquidity(values, pages, metrics, periods, thresholds):
    min_ratio = thresholds["current_ratio_min"]
    flags = []

    for period in periods:
        ratio = metrics.get((period, "current_ratio"))

        if ratio is not None and ratio < min_ratio:
            severity = "high" if ratio < 0.8 else "medium"
            flags.append(
                {
                    "rule": "liquidity",
                    "severity": severity,
                    "explanation": (
                        f"Current ratio of {round(ratio, 2)} in {period} "
                        f"is below {min_ratio}."
                    ),
                    "evidence": _evidence(
                        values,
                        pages,
                        [(period, "total_current_assets"), (period, "total_current_liabilities")],
                    ),
                }
            )

        st_debt = values.get((period, "short_term_debt"))
        cash = values.get((period, "cash_and_equivalents"))
        ocf = values.get((period, "operating_cash_flow"))

        if st_debt is not None and cash is not None and ocf is not None:
            if st_debt > cash + ocf:
                flags.append(
                    {
                        "rule": "liquidity",
                        "severity": "high",
                        "explanation": (
                            f"Short-term debt in {period} exceeds cash plus "
                            f"operating cash flow."
                        ),
                        "evidence": _evidence(
                            values,
                            pages,
                            [
                                (period, "short_term_debt"),
                                (period, "cash_and_equivalents"),
                                (period, "operating_cash_flow"),
                            ],
                        ),
                    }
                )

    return flags


def rule_customer_concentration(values, pages, metrics, periods, thresholds):
    max_pct = thresholds["customer_concentration_max_pct"]
    flags = []

    for period in periods:
        pct = values.get((period, "top_customer_revenue_pct"))

        if pct is None:
            continue

        if pct > max_pct:
            severity = "high" if pct >= 2 * max_pct else "medium"
            flags.append(
                {
                    "rule": "customer_concentration",
                    "severity": severity,
                    "explanation": (
                        f"Top customer accounted for {pct}% of revenue in "
                        f"{period} (threshold {max_pct}%)."
                    ),
                    "evidence": _evidence(
                        values, pages, [(period, "top_customer_revenue_pct")]
                    ),
                }
            )

    return flags


def run_rules(facts, thresholds=None):
    """Run all active rules on extracted facts. Returns (flags, skipped)."""
    thresholds = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    values, pages, metrics, periods = _build_lookups(facts)

    flags = []
    flags += rule_revenue_decline(values, pages, metrics, periods, thresholds)
    flags += rule_margin_compression(values, pages, metrics, periods, thresholds)
    flags += rule_leverage(values, pages, metrics, periods, thresholds)
    flags += rule_liquidity(values, pages, metrics, periods, thresholds)
    flags += rule_customer_concentration(values, pages, metrics, periods, thresholds)

    flags.sort(key=lambda flag: SEVERITY_ORDER[flag["severity"]])

    skipped = [
        {"rule": name, "severity": "skipped", "explanation": reason, "evidence": []}
        for name, reason in DORMANT_RULES
    ]

    return flags, skipped