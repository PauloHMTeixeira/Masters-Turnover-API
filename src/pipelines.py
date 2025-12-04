import pandas as pd
from beartype import beartype
import os

from src.utils import (
    data_pipeline,
    ml_pipeline,
    generate_cover_pdf,
    merge_pdfs,
)


@beartype
def main(dados: dict):
    """
    Função main onde as pipelines de dados e de Machine Learning vão ser executadas.

    Args:
        dados (dict): Dicionário onde estão os dados vindo da requisição.

    Returns:
        response_dict (dict): Dicionário com os endereços onde os resultado vão ficar 
            disponibilizados.
    """
    path = dados.get("path")
    file_name = dados.get("file_name")
    turnover_col = dados.get("turnover_col")

    df = pd.read_csv(path)

    # Ajuste: criar coluna 'Attrition'
    if turnover_col not in df.columns:
        raise ValueError(f"A coluna '{turnover_col}' não existe no dataset.")

    df["Attrition"] = df[turnover_col]

    df_ml = df.copy()

    # Gera PDFs individuais
    pdf_data = data_pipeline(df, file_name)
    pdf_ml = ml_pipeline(df_ml, file_name)

    # Gera capa
    cover_path = f"{file_name}_COVER.pdf"
    generate_cover_pdf(cover_path, file_name, turnover_col)

    # PDF final
    final_path = f"{file_name}_RELATORIO_FINAL.pdf"

    merge_pdfs(final_path, [cover_path, pdf_data, pdf_ml])

    return {
        "Final Report": final_path,
        "Data Analysis Path": pdf_data,
        "Feature Importance Path": pdf_ml,
        "Cover Path": cover_path,
    }
