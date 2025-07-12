import logging
from flask import Flask, request, jsonify

from src.pipelines import main
from src.validators import validar_entrada


app = Flask(__name__)

@app.route('/health-check', methods=['GET'])
def test():
    """
    Função para teste de rota, apenas para verificar se está funcionando.

    Args:
        None
    
    Returns:
        None
    """
    return jsonify({'message': 'Teste feito!'})

@app.route('/processa', methods=['POST'])
def processa():
    """
    Função principal da pipeline de dados e de ML.

    Args:
        None
    
    Returns:
        None
    """
    try:
        dados = request.get_json(force=True)

        logging.info("Requisição recebida %s", dados)

        erros = validar_entrada(dados)
        if erros:
            logging.warning("Foram encontrados problemas na requisição enviada: %s", erros)
            return jsonify({"erro": "Entrada inválida", "detalhes": erros}), 400

        resultado = main(dados)

        logging.info("Processamento dos dados concluido.")

        return jsonify(resultado)
    except Exception as e:
        logging.exception("Erro durante processamento")
        return jsonify({'erro': 'Erro interno', 'mensagem': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=app.config.get("DEBUG", True))