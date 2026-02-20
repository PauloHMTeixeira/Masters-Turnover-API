# src/pdf_hr_summary.py
from __future__ import annotations

import os
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def _register_font_if_available() -> str:
    """Registra uma fonte TrueType (para acentos) se existir localmente.

    Returns:
        Nome da fonte registrada (ou Helvetica como fallback).
    """
    # Tenta fontes comuns (Windows/Linux). Ajuste se quiser.
    candidates = [
        ("DejaVuSans", "DejaVuSans.ttf"),
        ("Arial", "arial.ttf"),
        ("Calibri", "calibri.ttf"),
    ]

    for font_name, font_file in candidates:
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_file))
            return font_name
        except Exception:
            continue

    return "Helvetica"


def generate_hr_summary_pdf(
    output_path: str,
    file_name: str,
    hr_summary_text: str,
    *,
    subtitle: Optional[str] = None,
) -> str:
    """Gera um PDF com o resumo interpretativo para RH.

    A página é pensada para ser inserida logo após a capa do relatório final.

    Args:
        output_path: Caminho de saída do PDF (ex.: data/<file>_HR_SUMMARY.pdf).
        file_name: Identificador do relatório (usado como título secundário).
        hr_summary_text: Texto já pronto (gerado pela OpenAI).
        subtitle: Subtítulo opcional.

    Returns:
        Caminho do PDF gerado.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    font_name = _register_font_if_available()

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    # Cabeçalho
    c.setFont(font_name, 16)
    c.drawString(2.2 * cm, height - 2.5 * cm, "Resumo interpretativo para RH")

    c.setFont(font_name, 10)
    c.drawString(2.2 * cm, height - 3.1 * cm, f"Relatório: {file_name}")

    if subtitle:
        c.setFont(font_name, 10)
        c.drawString(2.2 * cm, height - 3.6 * cm, subtitle)

    # Corpo em parágrafos com quebra automática
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=14,
        spaceAfter=8,
    )

    # Frame para texto
    frame = Frame(
        2.2 * cm,
        2.0 * cm,
        width - 4.4 * cm,
        height - 6.0 * cm,
        showBoundary=0,
    )

    # Converte quebras de linha em <br/>
    safe_text = hr_summary_text.strip().replace("\n", "<br/>")
    story = [Paragraph(safe_text, body_style)]

    frame.addFromList(story, c)

    # Rodapé simples
    c.setFont(font_name, 8.5)
    c.drawRightString(width - 2.2 * cm, 1.3 * cm, "Gerado automaticamente – usar como apoio à interpretação")

    c.showPage()
    c.save()

    return output_path