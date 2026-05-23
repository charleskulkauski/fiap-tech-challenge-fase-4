"""Exporta o relatório técnico Markdown para PDF (entrega individual, fora do pipeline)."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from textwrap import wrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MD_PATH = PROJECT_ROOT / "docs" / "RELATORIO_TECNICO_TECH_CHALLENGE_FASE4.md"
PDF_PATH = PROJECT_ROOT / "docs" / "RELATORIO_TECNICO_TECH_CHALLENGE_FASE4.pdf"


def _strip_md_inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text.strip()


def _markdown_to_plain_text(md: str) -> str:
    lines_out: list[str] = []
    in_code = False

    for raw in md.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if stripped == "---":
            lines_out.extend(["", "=" * 78, ""])
            continue
        if in_code:
            lines_out.append(line)
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= {"-", ":"} for c in cells):
                continue
            lines_out.append(" | ".join(_strip_md_inline(c) for c in cells))
            continue

        if line.startswith("# "):
            lines_out.extend(["", _strip_md_inline(line[2:]).upper(), "-" * 72, ""])
            continue
        if line.startswith("## "):
            lines_out.extend(["", _strip_md_inline(line[3:]), "-" * 48, ""])
            continue
        if line.startswith("### "):
            lines_out.append("")
            lines_out.append(_strip_md_inline(line[4:]))
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            lines_out.append(f"  • {_strip_md_inline(stripped[2:])}")
            continue
        if re.match(r"^\d+\.\s", stripped):
            lines_out.append(f"  {_strip_md_inline(stripped)}")
            continue

        if stripped:
            lines_out.append(_strip_md_inline(stripped))
        else:
            lines_out.append("")

    normalized: list[str] = []
    blank = False
    for line in lines_out:
        is_blank = not line.strip()
        if is_blank and blank:
            continue
        normalized.append(line)
        blank = is_blank
    return "\n".join(normalized).strip() + "\n"


def _export_pdf(text: str, output_path: Path) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = A4
    margin_x = 45
    margin_top = 50
    margin_bottom = 45
    line_height = 13
    max_chars = 98

    c = canvas.Canvas(str(output_path), pagesize=A4)
    y = page_h - margin_top
    page_num = 1

    def new_page() -> None:
        nonlocal y, page_num
        c.setFont("Helvetica", 8)
        c.drawRightString(page_w - margin_x, 28, f"Página {page_num}")
        c.showPage()
        page_num += 1
        y = page_h - margin_top

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin_x, y, "Relatório Técnico — Tech Challenge Fase 4")
    y -= 22

    for raw_line in text.splitlines():
        logical = raw_line if raw_line.strip() else " "
        is_heading = (
            len(raw_line) > 0
            and raw_line == raw_line.upper()
            and not raw_line.startswith(" ")
            and not raw_line.startswith("•")
            and not raw_line.startswith("  ")
            and "|" not in raw_line
            and len(raw_line) < 80
            and raw_line.strip("-") != ""
        )
        is_subheading = (
            raw_line.endswith("-")
            and len(raw_line) >= 10
            and set(raw_line.strip("-")) == {"-"}
        )

        if is_subheading:
            continue

        font = "Helvetica-Bold" if is_heading else "Helvetica"
        size = 11 if is_heading else 9
        c.setFont(font, size)

        wrapped = wrap(
            logical,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [" "]

        for segment in wrapped:
            if y < margin_bottom:
                new_page()
                c.setFont(font, size)
            c.drawString(margin_x, y, segment[:120])
            y -= line_height + (2 if is_heading else 0)

    c.setFont("Helvetica", 8)
    c.drawRightString(page_w - margin_x, 28, f"Página {page_num}")
    c.save()
    return output_path


def main() -> int:
    if not MD_PATH.is_file():
        print(f"Arquivo não encontrado: {MD_PATH}")
        return 1

    md_text = MD_PATH.read_text(encoding="utf-8")
    plain = _markdown_to_plain_text(md_text)
    pdf_path = _export_pdf(plain, PDF_PATH)
    print(f"PDF gerado: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
