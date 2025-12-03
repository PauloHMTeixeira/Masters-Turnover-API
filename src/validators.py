from beartype import beartype

@beartype
def validar_entrada(dados: dict):
    """
    Valida o dicionário de entrada. Retorna lista de erros, se houver.
    """
    erros = []

    # Validar path
    if 'path' not in dados or not isinstance(dados['path'], str):
        erros.append("Campo 'path' é obrigatório e deve ser string.")

    # Validar file_name
    if 'file_name' not in dados or not isinstance(dados['file_name'], str):
        erros.append("Campo 'file_name' é obrigatório e deve ser string.")

    # Novo campo necessário para a interface
    if 'turnover_col' not in dados or not isinstance(dados['turnover_col'], str):
        erros.append("Campo 'turnover_col' é obrigatório e deve ser string.")

    return erros
