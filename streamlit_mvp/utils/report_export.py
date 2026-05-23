from __future__ import annotations

from datetime import datetime
from pathlib import Path
from textwrap import wrap


def build_patient_acknowledgment_section(
    *,
    acknowledged_non_diagnostic: bool,
    acknowledged_reviewed: bool,
    acknowledged_at: datetime | str | None = None,
    case_id: str | None = None,
) -> str:
    if isinstance(acknowledged_at, datetime):
        recorded_at = acknowledged_at.isoformat(timespec="seconds")
    elif isinstance(acknowledged_at, str) and acknowledged_at.strip():
        recorded_at = acknowledged_at.strip()
    else:
        recorded_at = datetime.now().isoformat(timespec="seconds")

    non_diag_mark = "[x]" if acknowledged_non_diagnostic else "[ ]"
    reviewed_mark = "[x]" if acknowledged_reviewed else "[ ]"

    lines = [
        "## Declaração de ciência",
        "",
        "Registro de confirmação marcada pelo paciente/usuário responsável "
        "no momento da exportação do relatório:",
        "",
        f"- {non_diag_mark} Confirmo que o conteúdo é apoio não diagnóstico.",
        f"- {reviewed_mark} Confirmo que os dados foram revisados.",
        "",
        f"**Registrado em:** {recorded_at}",
    ]
    if case_id:
        lines.append(f"**Caso:** {case_id}")
    return "\n".join(lines)


def build_final_report_markdown(
    base_md: str,
    complemento_medico: str,
    *,
    acknowledgment: dict | None = None,
) -> str:
    parts = [base_md.rstrip()]

    extra = (complemento_medico or "").strip()
    if extra:
        parts.append("\n\n---\n\n## Complemento do médico\n\n" + extra)

    if acknowledgment:
        section = build_patient_acknowledgment_section(
            acknowledged_non_diagnostic=bool(
                acknowledgment.get("acknowledged_non_diagnostic", False)
            ),
            acknowledged_reviewed=bool(
                acknowledgment.get("acknowledged_reviewed", False)
            ),
            acknowledged_at=acknowledgment.get("acknowledged_at"),
            case_id=acknowledgment.get("case_id"),
        )
        parts.append("\n\n---\n\n" + section)

    return "".join(parts) + "\n"


def _markdown_to_plain_text(text: str) -> str:
    lines_out: list[str] = []
    in_code_block = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if stripped == "```":
            in_code_block = not in_code_block
            continue
        if stripped == "---":
            lines_out.append("")
            lines_out.append("=" * 72)
            lines_out.append("")
            continue

        line = raw
        if not in_code_block:
            hash_count = len(line) - len(line.lstrip("#"))
            if hash_count > 0:
                line = line[hash_count:].lstrip()
            if line.startswith("- "):
                line = f"* {line[2:]}"
            elif line.startswith("* "):
                line = f"* {line[2:]}"

        line = line.replace("`", "")
        lines_out.append(line)

    normalized: list[str] = []
    previous_blank = False
    for line in lines_out:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = is_blank
    return "\n".join(normalized).strip() + "\n"


def export_pdf_from_text(text: str, output_path: Path) -> Path:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError(
            "Dependência 'reportlab' não encontrada. "
            "Instale com: pip install reportlab"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_text = _markdown_to_plain_text(text)
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    margin_x = 48
    line_width_chars = 105
    y = height - 48

    c.setFont("Helvetica", 10)
    for raw_line in pdf_text.splitlines():
        logical_line = raw_line.rstrip() if raw_line else " "
        wrapped_lines = wrap(
            logical_line,
            width=line_width_chars,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [" "]

        for line in wrapped_lines:
            if y < 48:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 48
            c.drawString(margin_x, y, line)
            y -= 14

    c.save()
    return output_path
