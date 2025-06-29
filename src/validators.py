from beartype import beartype

@beartype
def validar_entrada(dados: dict):
    """
    Valida o dicionário de entrada. Retorna lista de erros, se houver.

    Args:
        dados (dict): JSON da requisição

    Returns:
        list: Lista de strings com erros encontrados
    """
    erros = []

    if 'path' not in dados and not isinstance(dados['path'], str):
        erros.append("Campo 'path' é obrigatório.")

    return erros
