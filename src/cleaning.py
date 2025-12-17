import pandas as pd
import numpy as np
import re

def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.60) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove colunas com proporção de missing acima de threshold.
    Retorna (df_filtrado, colunas_removidas).
    """
    missing_ratio = df.isna().mean()
    to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    return df.drop(columns=to_drop, errors="ignore"), to_drop


def drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove colunas com apenas 1 valor único (ignorando NaN).
    """
    to_drop = []
    for col in df.columns:
        nun = df[col].dropna().nunique()
        if nun <= 1:
            to_drop.append(col)
    return df.drop(columns=to_drop, errors="ignore"), to_drop


def drop_high_cardinality_columns(df: pd.DataFrame, threshold_ratio: float = 0.95, min_rows: int = 50) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove colunas com cardinalidade muito alta (quase ID), usando razão:
      nunique / n_rows > threshold_ratio
    Só aplica se n_rows >= min_rows para evitar falso positivo em dataset pequeno.
    """
    n = len(df)
    if n < min_rows:
        return df, []

    to_drop = []
    for col in df.columns:
        # ignora numéricas contínuas (muitas vezes terão alta cardinalidade, mas não são IDs)
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        nunique = df[col].nunique(dropna=True)
        if n > 0 and (nunique / n) > threshold_ratio:
            to_drop.append(col)

    return df.drop(columns=to_drop, errors="ignore"), to_drop


def add_date_features_and_drop_original(df: pd.DataFrame, max_date_cols: int = 10) -> tuple[pd.DataFrame, list[str]]:
    """
    Detecta colunas que parecem data (por nome ou por parse bem-sucedido),
    cria features ano/mês e remove a coluna original.
    Versão mínima e segura.
    """
    df = df.copy()
    dropped = []
    date_like_cols = []

    # Heurística por nome
    name_patterns = re.compile(r"(date|data|dt|admission|hire|termination|resignation|start|end)", re.IGNORECASE)

    for col in df.columns:
        if len(date_like_cols) >= max_date_cols:
            break

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_like_cols.append(col)
            continue

        if name_patterns.search(col):
            date_like_cols.append(col)
            continue

        # tentativa leve de parse (só se for object e poucos valores únicos para reduzir custo)
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(200)
            if sample.empty:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
                if parsed.notna().mean() > 0.8:
                    date_like_cols.append(col)
            except Exception:
                pass

    # Criar features e remover original
    for col in date_like_cols:
        try:
            parsed_full = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            df[f"{col}__year"] = parsed_full.dt.year
            df[f"{col}__month"] = parsed_full.dt.month
            dropped.append(col)
        except Exception:
            # se não deu pra parsear tudo, ignora
            continue

    df = df.drop(columns=dropped, errors="ignore")
    return df, dropped


def clean_dataset_generic(
    df: pd.DataFrame,
    target_col: str = "Attrition",
    missing_threshold: float = 0.60,
    high_card_ratio: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    """
    Orquestra limpeza universal, protegendo target_col.
    Retorna df_limpo e um 'report' com o que foi removido/criado.
    """
    df = df.copy()
    report = {
        "dropped_missing": [],
        "dropped_constant": [],
        "dropped_high_cardinality": [],
        "date_cols_processed": [],
    }

    # Nunca remover o target
    protected = {target_col}

    # 1) Missing alto
    df_tmp, dropped = drop_high_missing_columns(df.drop(columns=list(protected), errors="ignore"), threshold=missing_threshold)
    report["dropped_missing"] = dropped
    df = pd.concat([df_tmp, df[list(protected)]], axis=1) if target_col in df.columns else df_tmp

    # 2) Constantes
    df_tmp, dropped = drop_constant_columns(df.drop(columns=list(protected), errors="ignore"))
    report["dropped_constant"] = dropped
    df = pd.concat([df_tmp, df[list(protected)]], axis=1) if target_col in df.columns else df_tmp

    # 3) Alta cardinalidade (IDs disfarçados)
    df_tmp, dropped = drop_high_cardinality_columns(df.drop(columns=list(protected), errors="ignore"), threshold_ratio=high_card_ratio)
    report["dropped_high_cardinality"] = dropped
    df = pd.concat([df_tmp, df[list(protected)]], axis=1) if target_col in df.columns else df_tmp

    # 4) Datas -> features
    df_tmp, dropped = add_date_features_and_drop_original(df.drop(columns=list(protected), errors="ignore"))
    report["date_cols_processed"] = dropped
    df = pd.concat([df_tmp, df[list(protected)]], axis=1) if target_col in df.columns else df_tmp

    return df, report