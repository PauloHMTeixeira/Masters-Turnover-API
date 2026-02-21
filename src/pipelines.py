import pandas as pd
from beartype import beartype
import os

from src.cleaning import clean_dataset_generic
from src.pdf_hr_summary import generate_hr_summary_pdf
from src.utils import (
    data_pipeline,
    ml_pipeline,
    prepare_attrition_column,
    generate_cover_pdf,
    merge_pdfs,
)


@beartype
def main(dados: dict):
    """
    Executa as pipelines de dados e ML, gera capa e unifica os PDFs em um relatório final.

    Args:
        dados (dict): Dicionário com 'path', 'file_name', 'turnover_col'.

    Returns:
        dict: Paths dos artefatos gerados.
    """
    path = dados.get("path")
    file_name = dados.get("file_name")
    turnover_col = dados.get("turnover_col")

    df = pd.read_csv(path)

    # -----------------------------
    # PASSO 1 — Target robusto
    # -----------------------------
    df = prepare_attrition_column(df, turnover_col)

    # -----------------------------
    # Pipeline de dados (EDA) - mantém df completo
    # -----------------------------
    pdf_data = data_pipeline(df.copy(), file_name)

    # -----------------------------
    # PASSO 2 — Limpeza universal (para ML)
    # -----------------------------
    df_ml, _clean_report = clean_dataset_generic(df.copy(), target_col=turnover_col)

    # Pipeline de ML (já deve assumir Attrition 0/1)
    pdf_ml, hr_summary = ml_pipeline(df_ml, file_name, turnover_col)

    # -----------------------------
    # Capa + Merge (PDF final)
    # -----------------------------
    cover_path = os.path.join("data", f"{file_name}_COVER.pdf")
    final_path = os.path.join("data", f"{file_name}_RELATORIO_FINAL.pdf")

    hr_pdf_path = None
    if hr_summary:
        hr_pdf_path = os.path.join("data", f"{file_name}_HR_SUMMARY.pdf")
        generate_hr_summary_pdf(
            output_path=hr_pdf_path,
            file_name=file_name,
            hr_summary_text=hr_summary,
            subtitle="Leitura em linguagem simples para apoio à área de Recursos Humanos",
        )

    pdfs_to_merge = []

    pdfs_to_merge.append(cover_path)
    if hr_pdf_path:
        pdfs_to_merge.append(hr_pdf_path)
    pdfs_to_merge.append(pdf_data)
    pdfs_to_merge.append(pdf_ml)

    generate_cover_pdf(cover_path, file_name, turnover_col)
    merge_pdfs(final_path, pdfs_to_merge)

    return {
        "Final Report": final_path,
        "HR Summary": hr_summary,
        "Data Analysis Path": pdf_data,
        "Feature Importance Path": pdf_ml,
        "Cover Path": cover_path,
    }
