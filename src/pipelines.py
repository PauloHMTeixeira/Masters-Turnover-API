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
    file_name = dados.get("file_name")

    df = pd.read_csv(path)
    response_dict = {}

    df_ml = df.copy()
    df_ml_gender = df[df['Gender'] == 'Female'].copy()

    response_data = data_pipeline(df, file_name)
    response_ml = ml_pipeline(df_ml, file_name)
    response_ml_gender = ml_pipeline(df_ml_gender, file_name, True)

    response_dict['Data Analysis Path'] = response_data

    response_dict['Feature Importance Path'] = response_ml

    response_dict['Feature Importance Gender Path'] = response_ml_gender

    return response_dict
