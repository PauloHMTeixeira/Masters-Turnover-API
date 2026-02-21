# src/llm_summary.py
from __future__ import annotations

import os
from typing import Dict, List, Optional

from openai import OpenAI


def generate_hr_summary(
    *,
    dataset_label: str,
    target_col_original: str,
    n_rows: int,
    n_cols: int,
    class_balance: Optional[Dict[str, int]],
    model_selected: str,
    f1_holdout: float,
    top_features: List[str],
    notes: Optional[str] = None,
    model: str = "gpt-4.1-mini",
) -> str:
    """Gera um resumo em linguagem acessível para RH com base nos resultados do pipeline.

    Args:
        dataset_label: Rótulo amigável do dataset (ex.: nome do arquivo).
        target_col_original: Nome original da coluna alvo informada pelo usuário (ex.: LeaveOrNot).
        n_rows: Número de linhas do dataset.
        n_cols: Número de colunas do dataset (incluindo a alvo original, se aplicável).
        class_balance: Contagem por classe, ex.: {"nao": 3053, "sim": 1600}.
        model_selected: Modelo final selecionado (ex.: Gradient Boosting).
        f1_holdout: F1-score no conjunto de teste (hold-out), após tuning.
        top_features: Lista com as variáveis mais relevantes (strings).
        notes: Observações opcionais (ex.: "dataset anonimizado").
        model: Modelo da OpenAI a ser usado.

    Returns:
        Texto resumido e interpretativo para público de RH.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não configurada no ambiente.")

    client = OpenAI(api_key=api_key)

    balance_txt = ""
    if class_balance:
        balance_txt = f"Distribuição do turnover (contagens): {class_balance}."

    notes_txt = f"Notas: {notes}." if notes else ""

    prompt = f"""
Você é um analista de People Analytics escrevendo para profissionais de RH (não técnicos).
Explique os resultados abaixo de forma simples e prática, evitando jargões.
Não invente fatos.

Contexto:
- Dataset: {dataset_label}
- Tamanho: {n_rows} linhas, {n_cols} colunas
- Coluna alvo (turnover): {target_col_original}
- {balance_txt}
- Modelo final selecionado: {model_selected}
- F1 no teste (hold-out): {f1_holdout:.2f}
- Principais variáveis (Top): {", ".join(top_features)}
- {notes_txt}

Entregue a resposta com:
1) Resumo executivo (3–5 linhas)
2) O que isso sugere (bullets)
3) Recomendações práticas para investigação (bullets)
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        text={"format": {"type": "text"}},
    )
    return resp.output_text.strip()