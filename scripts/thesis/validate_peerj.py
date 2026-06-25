"""Pre-submission validator for literature-review-peerj.docx.

Runs every PeerJ-relevant check from the audit plan in one go. Exits
non-zero on any failure so this can be wired into a pre-submission
gate.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
QN_P = f"{{{W}}}p"
QN_T = f"{{{W}}}t"

PEERJ_DOCX = Path("docs/research/literature-review-peerj.docx")
SUPP_S1 = Path("docs/research/literature-review-peerj.supp.S1.md")
PRISMA_PNG = Path("docs/research/figures/fig-peerj-prisma-flow.png")


REQUIRED_DECLARATION_HEADINGS = [
    "Funding",
    "Competing Interests",
    "Author Contributions",
    "Data Availability",
    "AI Use Declaration",
]


def para_text(p) -> str:
    return "".join((t.text or "") for t in p.findall(f".//{QN_T}"))


def main() -> int:
    if not PEERJ_DOCX.exists():
        print(f"FAIL: {PEERJ_DOCX} not found", file=sys.stderr)
        return 1

    doc = Document(str(PEERJ_DOCX))
    body = doc.element.body
    top_paras = list(body.iterchildren(QN_P))
    para_texts = [para_text(p) for p in top_paras]
    headings = {t.strip() for t in para_texts}
    full_text = "\n".join(para_texts)

    checks: list[tuple[str, bool, str]] = []

    # ---- A1: no guidance callouts -----------------------------------------
    n_callout = full_text.lower().count("remove this box")
    checks.append((
        "A1: no guidance callouts remain",
        n_callout == 0,
        f"found {n_callout} 'remove this box' marker(s)",
    ))

    # ---- A2: all 5 declaration headings present ---------------------------
    missing = [h for h in REQUIRED_DECLARATION_HEADINGS if h not in headings]
    checks.append((
        "A2: all 5 declarations present",
        not missing,
        f"missing: {missing or 'none'}",
    ))

    # ---- A3: first reference well-formed ----------------------------------
    chung_p = next((t for t in para_texts if t.startswith("Chung, H. W.")), None)
    chung_ok = (
        chung_p is not None
        and "et al." in chung_p
        and chung_p.count(",") < 10  # truncated, not 16-author spell-out
    )
    checks.append((
        "A3: Chung first reference truncated to 3+et al.",
        chung_ok,
        f"chung_p[:120] = {chung_p[:120] if chung_p else 'NOT FOUND'!r}",
    ))

    # ---- B1: no inconsistent citation forms remain ------------------------
    odd = []
    odd += re.findall(
        r"\([A-Z][A-Za-z\-]+,\s+[A-Z]\.,\s+[A-Z][A-Za-z\-]+\s+et al\.,\s+\d{4}[a-z]?\)",
        full_text,
    )
    odd += re.findall(
        r"\b[A-Z][A-Za-z\-]+,\s+[A-Z]\.,\s+[A-Z][A-Za-z\-]+\s+et al\.\s+\(\d{4}[a-z]?\)",
        full_text,
    )
    # 3+author ampersand form (excluding 2-author "A & B, YYYY")
    for m in re.findall(r"\([^)]*&[^)]*\d{4}[a-z]?\)", full_text):
        inside = m[1:-1]
        for part in inside.split(";"):
            if "&" not in part:
                continue
            before_amp = part.split("&")[0].strip()
            if before_amp.count(",") >= 1:
                odd.append(part.strip())
    checks.append((
        "B1: no malformed in-text citations",
        not odd,
        f"odd cites: {odd or 'none'}",
    ))

    # ---- B2: methodology ref count is internally consistent --------------
    # The bib was restored from 89 to 102 entries; methodology was updated to
    # "more than 100 references". Accept any of these formulations; only
    # reject the original "+90 references" pre-fix.
    has_old = "+90 references" in full_text
    has_count = any(
        marker in full_text
        for marker in ("89 references", "more than 100 references", "~100 references")
    )
    checks.append((
        "B2: Survey Methodology references count consistent",
        has_count and not has_old,
        f"has_count={has_count}, has_old +90={has_old}",
    ))

    # ---- B3: novelty paragraph present ------------------------------------
    has_novelty = any(
        t.startswith("Existing reviews in adjacent areas") for t in para_texts
    )
    checks.append((
        "B3: Intro novelty paragraph present",
        has_novelty,
        "",
    ))

    # ---- C1: PRISMA intro + caption present, NO embedded figures ---------
    # PeerJ requires figures to be uploaded as separate files, NOT embedded.
    # In-text reference and caption stay; the image lives on disk only.
    has_prisma_intro = any(t.startswith("Figure 3 summarises") for t in para_texts)
    has_prisma_caption = any(
        t.startswith("Figure 3. PRISMA-style flow") for t in para_texts
    )
    n_drawings = len(body.findall(f".//{{{W}}}drawing"))
    checks.append((
        "C1: PRISMA intro + caption present, no embedded figures (PeerJ rule)",
        has_prisma_intro and has_prisma_caption and n_drawings == 0,
        f"intro={has_prisma_intro}, caption={has_prisma_caption}, drawings={n_drawings}",
    ))

    # ---- C2: Acknowledgements not the placeholder -------------------------
    ack_idx = next(
        (i for i, t in enumerate(para_texts) if t.strip() == "Acknowledgements"),
        None,
    )
    ack_body = ""
    if ack_idx is not None:
        for t in para_texts[ack_idx + 1 :]:
            if t.strip():
                ack_body = t.strip()
                break
    ack_ok = bool(ack_body) and "has not declared" not in ack_body
    checks.append((
        "C2: Acknowledgements is filled (not placeholder)",
        ack_ok,
        f"ack_body[:80] = {ack_body[:80]!r}",
    ))

    # ---- D2: no straggler Figure/Table 2.x in prose -----------------------
    stragglers = re.findall(r"(Figure 2\.\d+|Table 2\.\d+)", full_text)
    checks.append((
        "D2: no Figure/Table 2.x stragglers in prose",
        not stragglers,
        f"stragglers: {sorted(set(stragglers)) or 'none'}",
    ))

    # ---- D3: every in-text Author-Year cite has a bibliography entry ------
    # Collect surnames cited in text: "(Surname, YYYY)" and "(Surname et al., YYYY)"
    # and narrative "Surname (YYYY)" and "Surname et al. (YYYY)".
    cited_authors = set()
    for m in re.finditer(
        r"\(([A-Z][A-Za-z\-]+)(?:\s+et al\.)?(?:\s+&\s+[A-Z][A-Za-z\-]+)?,\s+\d{4}[a-z]?\)",
        full_text,
    ):
        cited_authors.add(m.group(1))
    for m in re.finditer(
        r"\b([A-Z][A-Za-z\-]+)(?:\s+et al\.)?\s+\(\d{4}[a-z]?\)",
        full_text,
    ):
        cited_authors.add(m.group(1))
    # Find References section paragraphs.
    ref_idx = next(
        (i for i, t in enumerate(para_texts) if t.strip() == "References"),
        None,
    )
    ref_authors = set()
    if ref_idx is not None:
        for t in para_texts[ref_idx + 1 :]:
            if not t.strip():
                continue
            # Reference starts with "Surname, X." — pull the leading surname.
            m = re.match(r"^([A-Z][A-Za-z\-]+)", t.strip())
            if m:
                ref_authors.add(m.group(1))
    # Strip common false positives that look like author names but aren't.
    NON_AUTHOR = {
        "Figure", "Table", "Section", "Note", "Equation", "Appendix",
        "Chapter", "Source", "Anthropic", "OpenAI", "Google", "Meta",
        "Microsoft", "RQ1", "RQ2", "RQ3", "BLEU", "ROUGE", "NLP", "AI", "G",
        # Co-author surnames whose primary entry is under another surname.
        "Ghazanfar",  # Karami, Shemshaki, & Ghazanfar (2024)
        "Givi",       # Kirk & Givi (2025)
        "Gloor",      # Duerr & Gloor (2021)
        "Nissenbaum", # Susser, Roessler, & Nissenbaum (2019)
        "Schwartz",   # Braun & Schwartz (2025)
        "Stillwell",  # Matz, Kosinski, Nave, & Stillwell (2017)
    }
    missing_refs = sorted(cited_authors - ref_authors - NON_AUTHOR)
    # Filter out single-letter and short tokens
    missing_refs = [r for r in missing_refs if len(r) > 2]
    checks.append((
        "D3: every in-text citation has a bibliography entry",
        not missing_refs,
        f"missing: {missing_refs or 'none'}",
    ))

    # ---- line numbers (PeerJ peer-review requirement) ---------------------
    sectprs = body.findall(f".//{{{W}}}sectPr")
    has_lineno = any(sp.find(f".//{{{W}}}lnNumType") is not None for sp in sectprs)
    checks.append((
        "Line numbers enabled (PeerJ peer-review requirement)",
        has_lineno,
        "",
    ))

    # ---- word count in expected range -------------------------------------
    words = re.findall(r"\b\w+\b", full_text)
    wc_ok = 11_000 <= len(words) <= 15_000
    checks.append((
        f"Word count in expected range (got {len(words)})",
        wc_ok,
        "",
    ))

    # ---- supplementary file present ---------------------------------------
    checks.append((
        f"Supplementary S1 present at {SUPP_S1}",
        SUPP_S1.exists(),
        "",
    ))

    # ---- PRISMA source PNG present ----------------------------------------
    checks.append((
        f"PRISMA source PNG present at {PRISMA_PNG}",
        PRISMA_PNG.exists(),
        "",
    ))

    # ---- docx round-trips through python-docx -----------------------------
    try:
        Document(str(PEERJ_DOCX))
        roundtrip_ok = True
        roundtrip_detail = ""
    except Exception as exc:
        roundtrip_ok = False
        roundtrip_detail = f"exception: {exc}"
    checks.append((
        "docx round-trips through python-docx",
        roundtrip_ok,
        roundtrip_detail,
    ))

    # ---- core metadata set -------------------------------------------------
    cp = doc.core_properties
    meta_ok = (
        (cp.author or "").lower().startswith("osama")
        and (cp.last_modified_by or "").lower().startswith("osama")
        and bool(cp.title)
    )
    checks.append((
        "D1: core metadata set (author, lastModifiedBy, title)",
        meta_ok,
        f"author={cp.author!r}, lastModBy={cp.last_modified_by!r}, title={(cp.title or '')[:60]!r}",
    ))

    # ----- summary ---------------------------------------------------------
    print("=" * 78)
    print("PeerJ pre-submission validation")
    print("=" * 78)
    failed = 0
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {name}")
        if detail:
            print(f"         {detail}")
    print("=" * 78)
    print(f"Result: {len(checks) - failed}/{len(checks)} checks passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
