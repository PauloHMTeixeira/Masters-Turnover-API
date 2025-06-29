import logging

import pandas as pd
from beartype import beartype

from src.utils import data_pipeline, ml_pipeline


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
    numerical_ordinal = dados.get("numerical_ordinal")
    categorical_ordinal = dados.get("categorical_ordinal")
    categorical_nom = dados.get("categorical_nom")

    df = pd.read_csv(path)
    response_dict = {}

    response_data = data_pipeline(df)
    response_ml = ml_pipeline(df, numerical_ordinal, categorical_ordinal, categorical_nom)

    response_dict['Data Analysis Path'] = response_data

    response_dict['Feature Importance Path'] = response_ml

    return response_dict
