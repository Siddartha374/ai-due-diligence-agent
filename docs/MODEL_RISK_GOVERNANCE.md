
# Model Risk & Governance Note — AI Due Diligence Agent v1

## 1. Purpose and intended use

Decision-support tool for preliminary due diligence on public corporate filings.
Outputs are ranked risk flags and a cited narrative for human review. The system
is not a credit or investment decision system and makes no recommendations.

## 2. Governing design principles

- **Deterministic-first.** Every number that drives a conclusion is either
  extracted into a strict schema or computed in pure Python. The LLM never
  calculates.
- **Null discipline.** Absence of data is represented explicitly. Estimation and
  inference are prohibited by prompt and surfaced in the UI as honest nulls.
- **Traceability.** Every fact stores its source page and a verbatim snippet.
  Every narrative claim carries a citation tag. Every tag is audited.
- **Scope transparency.** Rules whose required inputs are not extracted are
  declared *dormant* in the UI, never silently skipped.

## 3. Controls inventory

| Stage | Control |
|---|---|
| Ingestion | SHA-256 hash, page count, deterministic-first classification with method recorded |
| Extraction | Strict JSON schema; null-if-absent instruction; per-fact page, snippet, confidence |
| Grounding | Snippet verified verbatim against the cited PDF page; failures displayed as unverified; click-to-inspect UI |
| Computation | Ratios and all risk rules are pure Python functions with a pytest suite |
| Reasoning | Citation tags mandatory; audit rejects invalid tags; regeneration with explicit correction; honest fallback warning if regeneration fails |
| Monitoring | Honest-null count and unverified-citation count displayed on every run; model name stored per fact for version tracking |

## 4. Mapping to SR 11-7 / NIST AI RMF

- **Effective challenge:** independent deterministic layers (rule engine,
  citation audit, snippet verification) challenge the LLM's output before display.
- **Documentation:** this note, the README, and in-code docstrings document
  purpose, design, thresholds, and limitations.
- **Ongoing monitoring:** verification rate and null rate are surfaced per run,
  giving operators a drift/quality signal.
- **Provider risk:** automatic retry with backoff on 429/503; model configurable
  via environment; behavior on provider failure is documented and honest.

## 5. Observed incidents and remediations

1. **Classifier front-matter bias.** An annual report was initially classified
   as a presentation because the LLM saw only cover pages. Remediation:
   full-document deterministic keyword scoring; LLM consulted only when
   confidence is low; informative sampling when consulted.
2. **Memory-fabricated citations.** The LLM cited page numbers that were never
   provided to it. Remediation: PDF-level snippet verification; unverified facts
   are flagged in the UI rather than displayed as trusted.

## 6. Limitations and roadmap

Transcript and contract extraction (and their dependent rules), industry-adjustable
thresholds, table-aware extraction, and production-grade persistence are deferred
to v2 and are declared in-app as dormant rules.

## 7. Data handling

Public documents only. API keys are stored locally in `.env` (never committed).
No intentional PII. Not investment advice.