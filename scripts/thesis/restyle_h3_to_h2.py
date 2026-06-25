"""Rewrite the Heading 3 style block in THESIS.docx so it visually matches Heading 2.

The thesis template's Heading 3 (styleId=882) is the Word default (Arial, teal,
14pt) while Heading 2 (styleId=945) has been customized to Times New Roman, bold,
12pt. The 56 inserted subheadings need to read as the same family as the parent
section headers, so we replace H3's pPr/rPr with H2's (keeping H3's
styleId/name/basedOn/next/link to preserve TOC hierarchy).

Run from repo root:

    uv run python scripts/thesis/restyle_h3_to_h2.py
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

THESIS = Path("docs/research/THESIS.docx")
TMP = Path("docs/research/THESIS.docx.tmp")

# Replacement pPr/rPr for Heading 3, mirroring Heading 2 (styleId=945) verbatim
# except for outlineLvl, which we keep at "2" (matches existing H2 in this doc).
NEW_PPR = (
    "<w:pPr>"
    "<w:pBdr></w:pBdr>"
    '<w:spacing/>'
    '<w:ind w:left="709"/>'
    '<w:outlineLvl w:val="2"/>'
    "</w:pPr>"
)
NEW_RPR = (
    "<w:rPr>"
    '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
    "<w:b/><w:bCs/>"
    '<w:sz w:val="24"/><w:szCs w:val="24"/>'
    '<w:lang w:val="en-US" w:eastAsia="en-US" w:bidi="ar-SA"/>'
    "</w:rPr>"
)


def main() -> None:
    with zipfile.ZipFile(THESIS) as z:
        styles = z.read("word/styles.xml").decode("utf-8")

    # Locate the Heading 3 style block (styleId=882) and rewrite its pPr + rPr.
    pattern = re.compile(
        r'(<w:style\b[^>]*?w:styleId="882"[^>]*>)(.*?)(</w:style>)',
        re.S,
    )
    m = pattern.search(styles)
    if not m:
        raise SystemExit("Could not locate styleId=882 (Heading 3) block")

    head, inner, tail = m.group(1), m.group(2), m.group(3)
    # Strip the existing pPr and rPr from the inner block.
    inner = re.sub(r"<w:pPr>.*?</w:pPr>", "", inner, count=1, flags=re.S)
    inner = re.sub(r"<w:rPr>.*?</w:rPr>", "", inner, count=1, flags=re.S)
    new_block = head + inner + NEW_PPR + NEW_RPR + tail

    new_styles = styles[: m.start()] + new_block + styles[m.end() :]

    # Repack: copy every entry from the original zip, swapping styles.xml.
    with zipfile.ZipFile(THESIS) as zin, zipfile.ZipFile(
        TMP, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                data = new_styles.encode("utf-8")
            zout.writestr(item, data)

    shutil.move(TMP, THESIS)
    print(f"Restyled Heading 3 to match Heading 2 in {THESIS}")


if __name__ == "__main__":
    main()
