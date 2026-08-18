from rules import run_rules


def make_fact(period, field, value, page=1):
    return {"period": period, "field": field, "value": value, "source_page": page}


def test_revenue_decline_flagged():
    facts = [
        make_fact("2023", "revenue", 100),
        make_fact("2024", "revenue", 90),
        make_fact("2025", "revenue", 80),
    ]
    flags, _ = run_rules(facts)
    assert "revenue_decline" in [f["rule"] for f in flags]


def test_no_flag_on_growth():
    facts = [
        make_fact("2023", "revenue", 100),
        make_fact("2024", "revenue", 110),
        make_fact("2025", "revenue", 120),
    ]
    flags, _ = run_rules(facts)
    assert "revenue_decline" not in [f["rule"] for f in flags]


def test_margin_compression_flagged():
    facts = [
        make_fact("2024", "revenue", 1000),
        make_fact("2024", "net_income", 200),
        make_fact("2025", "revenue", 1100),
        make_fact("2025", "net_income", 110),
    ]
    flags, _ = run_rules(facts)
    assert "margin_compression" in [f["rule"] for f in flags]


def test_leverage_flagged():
    facts = [
        make_fact("2025", "total_debt", 800),
        make_fact("2025", "ebitda", 100),
    ]
    flags, _ = run_rules(facts)
    assert "leverage" in [f["rule"] for f in flags]


def test_customer_concentration_flagged():
    facts = [make_fact("2025", "top_customer_revenue_pct", 18)]
    flags, _ = run_rules(facts)
    assert "customer_concentration" in [f["rule"] for f in flags]


def test_customer_concentration_not_flagged_below_threshold():
    facts = [make_fact("2025", "top_customer_revenue_pct", 8)]
    flags, _ = run_rules(facts)
    assert "customer_concentration" not in [f["rule"] for f in flags]


def test_missing_data_is_skipped_not_guessed():
    facts = [make_fact("2025", "revenue", 100)]
    flags, skipped = run_rules(facts)
    assert "guidance_credibility" in [s["rule"] for s in skipped]