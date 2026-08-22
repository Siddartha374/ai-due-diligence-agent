import hashlib
import html as html_lib
import json
import os
import re
import uuid
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect as sa_inspect,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from classification import classify_document
from extraction import (
    compute_derived_metrics,
    find_candidate_pages,
    run_extraction,
)
from reasoning import answer_question, generate_red_flag_report
from rules import run_rules

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir("/mount/src") or not os.access(BASE_DIR, os.W_OK):
    DATA_DIR = "/tmp"
else:
    DATA_DIR = BASE_DIR

STORAGE_DIR = os.path.join(DATA_DIR, "storage")
DB_PATH = os.path.join(DATA_DIR, "diligence.db")

os.makedirs(STORAGE_DIR, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


def _get_config(name, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default


api_key = _get_config("GEMINI_API_KEY")
model_name = _get_config("GEMINI_MODEL", "gemini-3.7-flash")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    filename = Column(String)
    stored_path = Column(String)
    doc_type = Column(String, default="unclassified")
    company_name = Column(String, nullable=True)
    classification_method = Column(String, nullable=True)
    page_count = Column(Integer, nullable=True)
    sha256 = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    pages = relationship(
        "Page",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class Page(Base):
    __tablename__ = "pages"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"))
    page_number = Column(Integer)
    text = Column(Text)

    document = relationship("Document", back_populates="pages")


class ExtractedFact(Base):
    __tablename__ = "extracted_facts"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"))
    statement = Column(String)
    field = Column(String)
    period = Column(String)
    value = Column(Float, nullable=True)
    unit = Column(String, nullable=True)
    currency = Column(String, nullable=True)
    source_page = Column(Integer, nullable=True)
    citation_valid = Column(Boolean, default=False)
    raw_snippet = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    model = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DerivedMetric(Base):
    __tablename__ = "derived_metrics"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"))
    period = Column(String)
    metric_name = Column(String)
    value = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RiskFlag(Base):
    __tablename__ = "risk_flags"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"))
    rule = Column(String)
    severity = Column(String)
    explanation = Column(Text)
    evidence_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RedFlagReport(Base):
    __tablename__ = "red_flag_reports"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"))
    markdown_content = Column(Text)
    hallucinated_citations = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(engine)


def apply_migrations():
    """Idempotent schema migration for databases created by older versions."""
    inspector = sa_inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("documents")}

    with engine.begin() as conn:
        if "company_name" not in columns:
            conn.exec_driver_sql(
                "ALTER TABLE documents ADD COLUMN company_name VARCHAR"
            )
        if "classification_method" not in columns:
            conn.exec_driver_sql(
                "ALTER TABLE documents ADD COLUMN classification_method VARCHAR"
            )


apply_migrations()


def extract_pdf_pages(file_path):
    pages = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((page_number, text))

    return pages


def fmt_number(value):
    if value is None or value != value:
        return "—"
    return f"{value:,.0f}"


def fmt_pct(value):
    if value is None or value != value:
        return "—"
    return f"{value * 100:.2f}%"


def verify_snippet(page_text, snippet):
    """Deterministic grounding check: does the cited snippet appear on the
    cited page? First an exact match ignoring punctuation and whitespace;
    then a lenient token match (at least 80% of tokens present, and every
    number with 3+ digits present) to tolerate light paraphrasing."""
    if not snippet or not page_text:
        return False

    norm_page = re.sub(r"[^a-z0-9]", "", page_text.lower())
    norm_snippet = re.sub(r"[^a-z0-9]", "", snippet.lower())

    if not norm_snippet:
        return False

    if norm_snippet in norm_page:
        return True

    page_tokens = set(re.findall(r"[a-z0-9]+", page_text.lower()))
    snippet_tokens = re.findall(r"[a-z0-9]+", snippet.lower())

    if not snippet_tokens:
        return False

    numbers = [t for t in snippet_tokens if t.isdigit() and len(t) >= 3]
    if numbers and not all(n in page_tokens for n in numbers):
        return False

    hits = sum(1 for t in snippet_tokens if t in page_tokens)
    return (hits / len(snippet_tokens)) >= 0.8


def render_highlighted_page(page_text, snippet):
    """Escape page text and wrap the cited snippet in a yellow highlight."""
    escaped = html_lib.escape(page_text or "")

    if not snippet:
        return f"<pre style='white-space: pre-wrap;'>{escaped}</pre>"

    needle = html_lib.escape(snippet)

    if needle in escaped:
        highlighted = escaped.replace(
            needle,
            f"<mark style='background-color:#ffe94d'>{needle}</mark>",
            1,
        )
        return f"<pre style='white-space: pre-wrap;'>{highlighted}</pre>"

    idx = escaped.lower().find(needle.lower())

    if idx >= 0:
        highlighted = (
            escaped[:idx]
            + f"<mark style='background-color:#ffe94d'>{escaped[idx:idx + len(needle)]}</mark>"
            + escaped[idx + len(needle):]
        )
        return f"<pre style='white-space: pre-wrap;'>{highlighted}</pre>"

    return (
        "<p><em>Snippet not located verbatim on this page — "
        "treat this citation as unverified.</em></p>"
        + f"<pre style='white-space: pre-wrap;'>{escaped}</pre>"
    )


st.set_page_config(
    page_title="AI Due Diligence Agent",
    page_icon="📄",
    layout="wide",
)

st.title("AI Due Diligence Agent")
st.caption(
    "Milestones 1A–1I: ingestion, classification, extraction with PDF-level "
    "citation verification, deterministic rules, audited reasoning, Q&A, "
    "company linking, visual analytics"
)

st.write(
    "Documents are classified before processing. Structured extraction and "
    "deterministic calculation happen before any LLM reasoning. Every fact's "
    "citation is verified against the PDF itself. The LLM narrates only "
    "pre-computed facts and flags, answers only from the structured store, "
    "and every claim is audited. No mock data."
)

st.subheader("Milestone 1A: Ingestion & classification")

company_name = st.text_input(
    "Company name (optional, used later to link reports, transcripts, contracts)"
)

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    st.write(f"Ready to ingest: **{uploaded_file.name}**")

    if st.button("Ingest PDF"):
        file_bytes = uploaded_file.getvalue()

        doc_id = uuid.uuid4().hex
        safe_name = os.path.basename(uploaded_file.name).replace(" ", "_")
        stored_path = os.path.join(STORAGE_DIR, f"{doc_id}_{safe_name}")

        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

        with open(stored_path, "wb") as f:
            f.write(file_bytes)

        try:
            pages = extract_pdf_pages(stored_path)
        except Exception as error:
            st.error(f"PDF text extraction failed: {error}")
            st.stop()

        doc_type, class_confidence, class_method = classify_document(
            pages, api_key, model_name
        )

        with SessionLocal() as session:
            document = Document(
                id=doc_id,
                filename=uploaded_file.name,
                stored_path=stored_path,
                doc_type=doc_type,
                company_name=company_name.strip() or None,
                classification_method=class_method,
                page_count=len(pages),
                sha256=sha256_hash,
            )

            session.add(document)

            for page_number, text in pages:
                session.add(
                    Page(
                        id=uuid.uuid4().hex,
                        document_id=doc_id,
                        page_number=page_number,
                        text=text,
                    )
                )

            session.commit()

        st.success(
            f"Ingested {uploaded_file.name}. {len(pages)} pages. "
            f"Classified as **{doc_type}** via {class_method} method."
        )

st.divider()

st.subheader("Documents in database")

with SessionLocal() as session:
    documents = (
        session.query(Document)
        .order_by(Document.created_at.desc())
        .all()
    )

if not documents:
    st.info("No documents ingested yet.")
    document_options = {}
else:
    document_rows = [
        {
            "company": document.company_name or "—",
            "filename": document.filename,
            "doc_type": document.doc_type,
            "method": document.classification_method or "—",
            "pages": document.page_count,
            "created_at": document.created_at,
        }
        for document in documents
    ]

    st.dataframe(pd.DataFrame(document_rows))

    if st.button("Reclassify all documents (1F)"):
        with SessionLocal() as session:
            all_documents = session.query(Document).all()

            for doc in all_documents:
                page_rows = (
                    session.query(Page)
                    .filter(Page.document_id == doc.id)
                    .order_by(Page.page_number)
                    .all()
                )

                new_type, conf, method = classify_document(
                    [(p.page_number, p.text) for p in page_rows],
                    api_key,
                    model_name,
                )

                doc.doc_type = new_type
                doc.classification_method = method

            session.commit()

        st.success("Classification complete.")
        st.rerun()

    document_options = {
        f"{document.filename} | {document.id[:8]}": document.id
        for document in documents
    }

    st.write("#### Link a company name")

    link_label = st.selectbox(
        "Select document to label",
        list(document_options.keys()),
        key="company_label_doc",
    )

    new_company = st.text_input(
        "Company name for this document",
        key="company_name_input",
    )

    if st.button("Save company name"):
        with SessionLocal() as session:
            doc = (
                session.query(Document)
                .filter(Document.id == document_options[link_label])
                .first()
            )
            doc.company_name = new_company.strip() or None
            session.commit()

        st.success("Company name saved.")
        st.rerun()

saved_facts = []
saved_metrics = []
saved_flags = []

st.divider()

st.subheader("Milestone 1B: Structured extraction")

if not api_key:
    st.error(
        "No Gemini API key found. Create `.env` with GEMINI_API_KEY to enable "
        "extraction. Deterministic rules and analytics still work on any facts "
        "already extracted."
    )

if not document_options:
    st.info("Ingest a PDF first, then run extraction.")
else:
    selected_label = st.selectbox(
        "Select document",
        list(document_options.keys()),
        key="main_document",
    )

    selected_document_id = document_options[selected_label]
    selected_document = next(
        d for d in documents if d.id == selected_document_id
    )

    st.caption(
        f"Classified as: {selected_document.doc_type or 'unclassified'} "
        f"(via {selected_document.classification_method or 'n/a'})"
    )

    if (selected_document.doc_type or "") != "financial_statement":
        st.warning(
            "This document is not classified as a financial statement. The "
            "extraction schema targets financial statements; expect sparse "
            "results or an honest stop."
        )

    if api_key and st.button("Run structured extraction (1B)"):
        with SessionLocal() as session:
            page_records = (
                session.query(Page)
                .filter(Page.document_id == selected_document_id)
                .order_by(Page.page_number)
                .all()
            )

            page_text_by_number = {
                page.page_number: page.text for page in page_records
            }

            candidate_pages = find_candidate_pages(
                [(page.page_number, page.text) for page in page_records]
            )

        if not candidate_pages:
            st.error(
                "No financial-statement pages detected in this document. "
                "Extraction stopped honestly instead of guessing."
            )
            st.stop()

        st.write(
            "Candidate pages selected deterministically: "
            f"{[page_number for page_number, _ in candidate_pages]}"
        )

        with st.spinner("Calling Gemini for structured extraction..."):
            try:
                parsed = run_extraction(candidate_pages, api_key, model_name)
            except Exception as error:
                st.error(f"Extraction failed: {error}")
                st.stop()

        facts = parsed.get("facts", [])
        currency = parsed.get("currency")
        unit = parsed.get("unit")

        with SessionLocal() as session:
            session.query(ExtractedFact).filter_by(
                document_id=selected_document_id
            ).delete()
            session.query(DerivedMetric).filter_by(
                document_id=selected_document_id
            ).delete()

            for fact in facts:
                session.add(
                    ExtractedFact(
                        id=uuid.uuid4().hex,
                        document_id=selected_document_id,
                        statement=fact.get("statement"),
                        field=fact.get("field"),
                        period=fact.get("period"),
                        value=fact.get("value"),
                        unit=unit,
                        currency=currency,
                        source_page=fact.get("page"),
                        citation_valid=verify_snippet(
                            page_text_by_number.get(fact.get("page")),
                            fact.get("snippet"),
                        ),
                        raw_snippet=fact.get("snippet"),
                        confidence=fact.get("confidence"),
                        model=model_name,
                    )
                )

            for metric_row in compute_derived_metrics(facts):
                period = metric_row.pop("period")

                for metric_name, metric_value in metric_row.items():
                    session.add(
                        DerivedMetric(
                            id=uuid.uuid4().hex,
                            document_id=selected_document_id,
                            period=period,
                            metric_name=metric_name,
                            value=metric_value,
                        )
                    )

            session.commit()

        st.success(
            f"Extraction complete. {len(facts)} facts stored. "
            f"Currency: {currency}. Unit: {unit}."
        )

    with SessionLocal() as session:
        saved_facts = (
            session.query(ExtractedFact)
            .filter(ExtractedFact.document_id == selected_document_id)
            .order_by(
                ExtractedFact.period,
                ExtractedFact.statement,
                ExtractedFact.field,
            )
            .all()
        )

        saved_metrics = (
            session.query(DerivedMetric)
            .filter(DerivedMetric.document_id == selected_document_id)
            .all()
        )

    if not saved_facts:
        st.info("No extracted facts yet for this document. Run extraction above.")
    else:
        st.write("### Extracted facts (stored in database)")

        fact_rows = [
            {
                "statement": fact.statement,
                "field": fact.field,
                "period": fact.period,
                "value": fact.value if fact.value is not None else "MISSING",
                "page": fact.source_page,
                "citation_ok": fact.citation_valid,
                "confidence": fact.confidence,
                "snippet": (fact.raw_snippet or "")[:120],
            }
            for fact in saved_facts
        ]

        facts_df = pd.DataFrame(fact_rows)

        selection_event = st.dataframe(
            facts_df,
            on_select="rerun",
            selection_mode="single-row",
        )

        st.caption(
            "Click any row to open its source page with the cited snippet "
            "highlighted. ✅ means the snippet was found verbatim on that page "
            "of the PDF."
        )

        selected_rows = selection_event.selection.rows

        if selected_rows:
            chosen_fact = saved_facts[selected_rows[0]]

            st.write(
                f"**Selected citation:** {chosen_fact.period} "
                f"{chosen_fact.field} = "
                f"{chosen_fact.value if chosen_fact.value is not None else 'MISSING'} "
                f"(page {chosen_fact.source_page})"
            )

            if chosen_fact.source_page is None:
                st.info("This fact has no page citation (value was MISSING).")
            else:
                with SessionLocal() as session:
                    source_page = (
                        session.query(Page)
                        .filter(Page.document_id == selected_document_id)
                        .filter(Page.page_number == chosen_fact.source_page)
                        .first()
                    )

                if source_page is None:
                    st.warning("Source page not found in database.")
                else:
                    st.write(f"### Source page {chosen_fact.source_page}")

                    st.markdown(
                        render_highlighted_page(
                            source_page.text, chosen_fact.raw_snippet
                        ),
                        unsafe_allow_html=True,
                    )

        missing_count = len(facts_df[facts_df["value"] == "MISSING"])
        stated_count = len(facts_df) - missing_count
        verified_count = len(
            facts_df[
                (facts_df["citation_ok"] == True)
                & (facts_df["value"] != "MISSING")
            ]
        )
        unverified_count = stated_count - verified_count

        verification_rate = (
            round(100 * verified_count / stated_count) if stated_count else 0
        )

        st.info(
            f"**Extraction quality panel** — stated facts extracted: "
            f"{stated_count} · fields not stated in this document (honest "
            f"nulls): {missing_count} · citations verified against the PDF: "
            f"{verified_count} of {stated_count} ({verification_rate}%) · "
            f"unverified: {unverified_count}. A high null count is normal: "
            f"the schema requests a full US-GAAP + Ind-AS checklist and most "
            f"filings state only part of it."
        )

        st.write("### Derived metrics (computed deterministically in Python)")

        metric_rows = [
            {
                "period": metric.period,
                "metric": metric.metric_name,
                "value": metric.value,
            }
            for metric in saved_metrics
        ]

        if metric_rows:
            metrics_df = pd.DataFrame(metric_rows)

            pivot = metrics_df.pivot_table(
                index="period",
                columns="metric",
                values="value",
                aggfunc="first",
            )

            st.dataframe(pivot)

st.divider()

st.subheader("Milestone 1C: Deterministic risk rules")

if document_options and saved_facts:
    if st.button("Run deterministic risk rules (1C)"):
        fact_dicts = [
            {
                "period": fact.period,
                "field": fact.field,
                "value": fact.value,
                "source_page": fact.source_page,
            }
            for fact in saved_facts
        ]

        flags, skipped = run_rules(fact_dicts)

        with SessionLocal() as session:
            session.query(RiskFlag).filter_by(
                document_id=selected_document_id
            ).delete()

            for flag in flags + skipped:
                session.add(
                    RiskFlag(
                        id=uuid.uuid4().hex,
                        document_id=selected_document_id,
                        rule=flag["rule"],
                        severity=flag["severity"],
                        explanation=flag["explanation"],
                        evidence_json=json.dumps(flag.get("evidence", [])),
                    )
                )

            session.commit()

        st.success(f"Risk engine complete. {len(flags)} active flags raised.")

    with SessionLocal() as session:
        saved_flags = (
            session.query(RiskFlag)
            .filter(RiskFlag.document_id == selected_document_id)
            .all()
        )

    if not saved_flags:
        st.info("No risk analysis yet for this document. Run the rules above.")
    else:
        active = [f for f in saved_flags if f.severity != "skipped"]
        dormant = [f for f in saved_flags if f.severity == "skipped"]

        if not active:
            st.success("No active red flags raised on the extracted data.")
        else:
            st.write("### Ranked red-flag report")

            for flag in active:
                if flag.severity == "high":
                    st.error(
                        f"**[{flag.severity.upper()}] {flag.rule}** — {flag.explanation}"
                    )
                elif flag.severity == "medium":
                    st.warning(
                        f"**[{flag.severity.upper()}] {flag.rule}** — {flag.explanation}"
                    )
                else:
                    st.info(
                        f"**[{flag.severity.upper()}] {flag.rule}** — {flag.explanation}"
                    )

                evidence = json.loads(flag.evidence_json or "[]")

                if evidence:
                    st.dataframe(pd.DataFrame(evidence))

        if dormant:
            st.write("### Dormant rules (waiting for more data types)")
            for flag in dormant:
                st.caption(f"{flag.rule}: {flag.explanation}")
elif document_options:
    st.info("Run extraction (1B) first, then run the risk rules.")

st.divider()

st.subheader("Milestone 1D: LLM reasoning & red-flag report")

if document_options and saved_facts and saved_flags:
    if st.button("Generate narrated report (1D)"):
        fact_dicts = [
            {
                "period": fact.period,
                "statement": fact.statement,
                "field": fact.field,
                "value": fact.value,
                "unit": fact.unit,
                "source_page": fact.source_page,
            }
            for fact in saved_facts
        ]

        metric_dicts = [
            {
                "period": metric.period,
                "metric_name": metric.metric_name,
                "value": metric.value,
            }
            for metric in saved_metrics
        ]

        flag_dicts = [
            {
                "rule": flag.rule,
                "severity": flag.severity,
                "explanation": flag.explanation,
            }
            for flag in saved_flags
        ]

        document_metadata = {
            "company_name": selected_document.company_name,
            "filename": selected_document.filename,
            "document_type": selected_document.doc_type,
            "page_count": selected_document.page_count,
        }

        with st.spinner("Generating institutional memo (with citation audit)..."):
            report_text, hallucinations = generate_red_flag_report(
                fact_dicts, metric_dicts, flag_dicts, api_key, model_name,
                metadata=document_metadata,
            )

        with SessionLocal() as session:
            session.query(RedFlagReport).filter_by(
                document_id=selected_document_id
            ).delete()

            session.add(
                RedFlagReport(
                    id=uuid.uuid4().hex,
                    document_id=selected_document_id,
                    markdown_content=report_text,
                    hallucinated_citations=json.dumps(list(hallucinations)),
                )
            )

            session.commit()

        st.success("Report generated.")

        if hallucinations:
            st.error(
                f"Citation audit: after regeneration, the LLM still used "
                f"{len(hallucinations)} invalid citations: {hallucinations}. "
                "Treat the report below as unverified."
            )

        st.write("### 📄 Institutional red-flag report")
        st.markdown(report_text)
elif document_options:
    st.info("Run extraction (1B) and risk rules (1C) first to generate the report.")

st.divider()

st.subheader("Milestone 1H: Q&A over the structured store")

if document_options and saved_facts:
    question = st.text_input(
        "Ask a question about this company",
        placeholder="What should concern us about this company?",
        key="qa_question",
    )

    if st.button("Ask (1H)"):
        if not question.strip():
            st.warning("Type a question first.")
        elif not api_key:
            st.error("Q&A requires the Gemini API key.")
        else:
            fact_dicts = [
                {
                    "period": fact.period,
                    "statement": fact.statement,
                    "field": fact.field,
                    "value": fact.value,
                    "unit": fact.unit,
                    "source_page": fact.source_page,
                }
                for fact in saved_facts
            ]

            metric_dicts = [
                {
                    "period": metric.period,
                    "metric_name": metric.metric_name,
                    "value": metric.value,
                }
                for metric in saved_metrics
            ]

            flag_dicts = [
                {
                    "rule": flag.rule,
                    "severity": flag.severity,
                    "explanation": flag.explanation,
                }
                for flag in saved_flags
            ]

            document_metadata = {
                "company_name": selected_document.company_name,
                "filename": selected_document.filename,
                "document_type": selected_document.doc_type,
                "page_count": selected_document.page_count,
            }

            with st.spinner("Answering strictly from the structured store..."):
                answer_text, qa_hallucinations = answer_question(
                    question, fact_dicts, metric_dicts, flag_dicts,
                    api_key, model_name, metadata=document_metadata,
                )

            if qa_hallucinations:
                st.error(
                    f"Citation audit: after regeneration, the answer still used "
                    f"{len(qa_hallucinations)} invalid citations: {qa_hallucinations}. "
                    "Treat the answer below as unverified."
                )

            st.markdown(answer_text)

            used_tags = sorted(
                {
                    (kind, int(number))
                    for kind, number in re.findall(
                        r"\[(Fact|Metric|Flag)-(\d+)\]", answer_text
                    )
                }
            )

            citation_rows = []

            for kind, index in used_tags:
                if kind == "Fact" and 1 <= index <= len(saved_facts):
                    f = saved_facts[index - 1]
                    citation_rows.append(
                        {
                            "ref": f"[{kind}-{index}]",
                            "source": (
                                f"{f.period} {f.field} = "
                                f"{f.value if f.value is not None else 'MISSING'} "
                                f"(page {f.source_page})"
                            ),
                        }
                    )
                elif kind == "Metric" and 1 <= index <= len(saved_metrics):
                    m = saved_metrics[index - 1]
                    citation_rows.append(
                        {
                            "ref": f"[{kind}-{index}]",
                            "source": f"{m.period} {m.metric_name} = {m.value}",
                        }
                    )
                elif kind == "Flag" and 1 <= index <= len(saved_flags):
                    fl = saved_flags[index - 1]
                    citation_rows.append(
                        {
                            "ref": f"[{kind}-{index}]",
                            "source": f"{fl.rule} ({fl.severity})",
                        }
                    )

            if citation_rows:
                st.write("#### Citations used in this answer")
                st.dataframe(pd.DataFrame(citation_rows))
elif document_options:
    st.info("Run extraction (1B) first, then ask questions.")

st.divider()

st.subheader("Milestone 1E: Visual analytics")

if not document_options:
    st.info("Ingest a PDF first, then view analytics.")
else:
    viz_label = st.selectbox(
        "Select document for analytics",
        list(document_options.keys()),
        key="viz_document",
    )

    viz_document_id = document_options[viz_label]

    with SessionLocal() as session:
        viz_facts = (
            session.query(ExtractedFact)
            .filter(ExtractedFact.document_id == viz_document_id)
            .all()
        )
        viz_metrics = (
            session.query(DerivedMetric)
            .filter(DerivedMetric.document_id == viz_document_id)
            .all()
        )
        viz_flags = (
            session.query(RiskFlag)
            .filter(RiskFlag.document_id == viz_document_id)
            .all()
        )

    fact_df = pd.DataFrame(
        [
            {"period": f.period, "field": f.field, "value": f.value}
            for f in viz_facts
            if f.value is not None
        ],
        columns=["period", "field", "value"],
    )

    metric_df = pd.DataFrame(
        [
            {"period": m.period, "metric": m.metric_name, "value": m.value}
            for m in viz_metrics
            if m.value is not None
        ],
        columns=["period", "metric", "value"],
    )

    active_flags = [f for f in viz_flags if f.severity != "skipped"]

    if fact_df.empty:
        st.info("No extracted facts to chart yet. Run extraction (1B) first.")
    else:
        periods = sorted(fact_df["period"].unique())
        latest = periods[-1]

        latest_values = (
            fact_df[fact_df["period"] == latest].set_index("field")["value"]
        )

        if not metric_df.empty:
            latest_metrics = (
                metric_df[metric_df["period"] == latest]
                .set_index("metric")["value"]
            )
        else:
            latest_metrics = pd.Series(dtype=float)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Revenue (latest)", fmt_number(latest_values.get("revenue")))
        k2.metric("Net income (latest)", fmt_number(latest_values.get("net_income")))
        k3.metric("Net margin (latest)", fmt_pct(latest_metrics.get("net_margin")))
        k4.metric("Active risk flags", str(len(active_flags)))

        trend_df = fact_df[
            fact_df["field"].isin(["revenue", "net_income", "operating_cash_flow"])
        ]

        if not trend_df.empty:
            st.write("### Performance trend")
            trend_pivot = trend_df.pivot_table(
                index="period", columns="field", values="value", aggfunc="first"
            )
            trend_pivot = trend_pivot.reindex(sorted(trend_pivot.index))
            st.line_chart(trend_pivot)

        margin_df = metric_df[
            metric_df["metric"].isin(["gross_margin", "ebitda_margin", "net_margin"])
        ]

        if not margin_df.empty:
            st.write("### Margin trend (%)")
            margin_pivot = margin_df.pivot_table(
                index="period", columns="metric", values="value", aggfunc="first"
            )
            margin_pivot = margin_pivot.reindex(sorted(margin_pivot.index)) * 100
            st.line_chart(margin_pivot)

        cash_df = fact_df[
            fact_df["field"].isin(["operating_cash_flow", "capex", "free_cash_flow"])
        ]

        if not cash_df.empty:
            st.write("### Cash flow")
            cash_pivot = cash_df.pivot_table(
                index="period", columns="field", values="value", aggfunc="first"
            )
            cash_pivot = cash_pivot.reindex(sorted(cash_pivot.index))
            st.bar_chart(cash_pivot)

        capital_df = fact_df[
            (fact_df["period"] == latest)
            & (fact_df["field"].isin(["total_equity", "total_debt"]))
        ]

        profit_df = fact_df[
            (fact_df["period"] == latest)
            & (fact_df["field"].isin(["net_income", "tax_expense", "interest_expense"]))
        ]

        if len(capital_df) >= 2:
            pie_df = capital_df
            pie_title = "Capital structure (latest period)"
        elif len(profit_df) >= 2:
            pie_df = profit_df
            pie_title = "Profit split: net income vs tax vs finance costs (latest period)"
        else:
            pie_df = None

        if pie_df is not None:
            st.write(f"### {pie_title}")

            pie = (
                alt.Chart(pie_df)
                .mark_arc(innerRadius=60)
                .encode(
                    theta=alt.Theta("value:Q"),
                    color=alt.Color("field:N"),
                    tooltip=["field", "value"],
                )
            )

            st.altair_chart(pie, use_container_width=True)
        else:
            st.caption(
                "Pie chart unavailable: not enough balance sheet or profit "
                "data extracted for the latest period."
            )