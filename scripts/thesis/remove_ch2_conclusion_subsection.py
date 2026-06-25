"""Reviewer feedback #15 — remove per-chapter "Conclusion" subsections.

Audit (2026-06-07) found only ONE literal "Conclusion" subsection outside the
dedicated CONCLUSION chapter (Chapter VI): the unnumbered "Conclusion" Heading-2
at the end of Chapter II (Literature Review). Chapters I/III/IV/V are already
clean; §4.6 "Synthesis and Limitations" was deliberately left in scope-out (it is
a Results synthesis, not a "Conclusion", and §4.6.3 Limitations is load-bearing).

That Ch II "Conclusion" subsection also hosts Table 2.6 (reviewer #11's
related-work comparison matrix) and its caption/note. Per the author's decision:

  - DELETE the "Conclusion" Heading-2.
  - DELETE the two synthesis paragraphs ("Three interlocking patterns…" and
    "These gaps—absent marketing LLMs…").
  - KEEP para "This survey has examined…" (it carries the "(Tables 2.3 and 2.6)"
    in-text callout — reviewer #5) plus the full Table 2.6 block (caption, table,
    note).

Once the heading is gone the surviving gap paragraph + table fall under §2.9
"Open Research Questions" as its closing material — no Word "Conclusion"
subsection remains in Chapter II.

Idempotent: re-running after the heading is gone is a no-op. Anchored by visible
text, never by paragraph index. python-docx + lxml only; no byte-slicing.

Run:  uv run python scripts/thesis/remove_ch2_conclusion_subsection.py
      uv run python scripts/thesis/remove_ch2_conclusion_subsection.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document

DOCX = Path("docs/research/THESIS.docx")

# Text anchors (match on stripped prefix / exact form). These are stable across
# index shifts; we scope every match to the Ch II range to avoid the Ch VI
# "Conclusion" headings and the TOC field-cache twin.
ANCHOR_2_9_5_BODY = "As discussed in Section 2.8, ethical and legal concerns"
ANCHOR_CHAPTER_III = "CHAPTER III"
HEADING_STYLE_ID = "945"  # Heading 2
CONCLUSION_HEADING_TEXT = "Conclusion"
DELETE_PROSE_PREFIXES = (
    "Three interlocking patterns cut across the reviewed literature",
    "These gaps—absent marketing LLMs",
)


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    if not DOCX.exists():
        print(f"ERROR: {DOCX} not found", file=sys.stderr)
        return 1

    doc = Document(str(DOCX))
    paras = doc.paragraphs

    # 1) Bound the Chapter II range: from §2.9.5 body to the CHAPTER III heading.
    start = None
    end = None
    for i, p in enumerate(paras):
        t = p.text.strip()
        if start is None and t.startswith(ANCHOR_2_9_5_BODY):
            start = i
        elif start is not None and t.startswith(ANCHOR_CHAPTER_III) and (
            (p.style.style_id if p.style else None) == "944"
        ):
            end = i
            break

    if start is None or end is None:
        print(
            "ERROR: could not bound the Ch II range "
            f"(start={start}, end={end}); aborting.",
            file=sys.stderr,
        )
        return 1

    window = list(range(start + 1, end))

    # 2) Collect deletion targets within the window.
    targets: list[tuple[int, str]] = []
    for i in window:
        p = paras[i]
        sid = p.style.style_id if p.style else None
        t = p.text.strip()
        if sid == HEADING_STYLE_ID and t == CONCLUSION_HEADING_TEXT:
            targets.append((i, f"[H2 heading] {t!r}"))
        elif any(t.startswith(pre) for pre in DELETE_PROSE_PREFIXES):
            targets.append((i, f"[prose] {t[:70]!r}…"))

    if not targets:
        print(
            "No-op: Ch II 'Conclusion' subsection already removed "
            "(no matching heading/prose in range). Idempotent exit."
        )
        return 0

    print(f"Ch II range: paras [{start}..{end}]  (CHAPTER III at {end})")
    print(f"DELETIONS ({len(targets)}):")
    for i, desc in targets:
        print(f"  - para {i}: {desc}")

    # Sanity: confirm the kept material survives (para 'This survey has examined…'
    # + the Table 2.6 caption) so we never silently drop the table/callout.
    kept_survey = any(
        paras[i].text.strip().startswith("This survey has examined") for i in window
    )
    kept_caption = any(
        paras[i].text.strip().startswith("Table 2.6") for i in window
    )
    print(
        f"KEPT (verified present): survey/callout paragraph={kept_survey}, "
        f"Table 2.6 caption={kept_caption}"
    )
    if not (kept_survey and kept_caption):
        print(
            "ERROR: expected kept material (survey paragraph + Table 2.6 caption) "
            "not found — aborting to avoid losing the table/callout.",
            file=sys.stderr,
        )
        return 1

    if dry_run:
        print("\n--dry-run: no changes written.")
        return 0

    # 3) Remove the target <w:p> elements from the tree.
    for i, _desc in targets:
        el = paras[i]._element
        el.getparent().remove(el)

    doc.save(str(DOCX))
    print(f"\nSaved {DOCX}.")
    print(
        "FOLLOW-UP: refresh the Table of Contents in OnlyOffice (Ctrl+A → F9) — "
        "the cached TOC still lists the deleted Ch II 'Conclusion' Heading-2."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
