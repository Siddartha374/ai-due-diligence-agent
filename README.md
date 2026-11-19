# AI Due Diligence Agent

** AI due diligence for annual reports — every claim cited, every number verifiable, every blind spot declared.**

The AI Due Diligence Agent reads a company's PDF filings the way a rigorous junior analyst would: it extracts explicitly stated facts into a structured store, computes ratios deterministically, runs a pure-Python risk-rule engine, and only then lets an LLM narrate the results — with every citation audited against the source PDF before you see it.

Built to demonstrate what responsible AI in finance looks like in working code, not in a slide deck.

## Core design principle

> **Structured extraction and deterministic calculation happen before any LLM reasoning.**
> The LLM explains; it never invents. Remove the LLM entirely and the system still produces a fact-checked extraction and a rule-based red-flag list.

## What it does

- **Ingests & classifies** PDFs (financial statement / contract / presentation / transcript) *before* processing — deterministic-first, LLM consulted only on low confidence.
- **Extracts structured financial facts** (US GAAP & Ind-AS) into a strict schema: value, unit, period, source page, confidence, verbatim snippet. Missing values are honest `null`s — never estimated.
- **Computes ratios deterministically in Python** (gross / EBITDA / net margins, debt/EBITDA, interest coverage, current ratio).
- **Runs a pure-Python risk-rule engine**: revenue decline, margin compression, leverage, liquidity, customer concentration — unit-tested, zero LLM calls.
- **Narrates with citations**: the LLM writes red-flag reports and answers Q&A strictly from the structured store; every claim carries a `[Fact-X]` / `[Metric-X]` / `[Flag-X]` tag.
- **Audits every citation**: invalid tags trigger rejection and regeneration with a correction; snippets are verified verbatim against the cited PDF page; anything that fails verification is displayed as unverified.
- **Declares its blind spots**: rules whose required data isn't extracted (e.g., guidance credibility needs transcripts) are listed as *dormant* — never silently skipped.
- **Visual analytics**: KPI cards, performance/margin/cash-flow charts, capital-structure donut.

## Architecture

```text
                   ┌───────────────────────────────┐
 PDF upload ──────▶│ 1 · Ingestion & classification │
                   └──────────────┬────────────────
                                  ▼
                   ┌───────────────────────────────┐
                   │ 2 · Structured extraction      │  LLM, strict JSON schema,
                   │   honest nulls + page citations│  verbatim snippets
                   └──────────────┬────────────────┘
                                  ▼
              ┌─────────────────────────────────────────┐
              │ 3 · Deterministic metrics + risk rules   │  pure Python, unit-tested
              └────────────────────┬────────────────────┘
                                  ▼
              ┌─────────────────────────────────────────┐
              │ 4 · LLM reasoning — cited, audited,      │  reject + regenerate on
              │    regenerated on audit failure          │  invalid citations
              └────────────────────┬────────────────────┘
                                  ▼
              ┌─────────────────────────────────────────┐
              │ 5 · Report · Q&A · click-to-cite · charts│
              └─────────────────────────────────────────┘
   SQLite persistence at every stage: documents, pages, facts,
   metrics, flags, reports
