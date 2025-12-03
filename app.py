import os
import logging
from flask import Flask, request, jsonify

from src.pipelines import main
from src.validators import validar_entrada

app = Flask(__name__)

DATA_FOLDER = "data"

# Garante a existência da pasta data/
os.makedirs(DATA_FOLDER, exist_ok=True)

@app.route('/health-check', methods=['GET'])
def test():
    return jsonify({"status": "ok"}), 200


@app.route('/executar-interface', methods=['POST'])
def executar_interface():
    try:
        # Verifica se o arquivo foi enviado
        if 'file' not in request.files:
            return jsonify({"erro": "Nenhum arquivo foi enviado."}), 400

        file = request.files['file']
        turnover_col = request.form.get("turnover_col")

        # Valida a coluna de turnover
        if not turnover_col:
            return jsonify({"erro": "O campo 'turnover_col' é obrigatório."}), 400

        # Verifica extensão do arquivo
        filename = file.filename
        if not filename.lower().endswith(".csv"):
            return jsonify({"erro": "Envie um arquivo CSV válido."}), 400

        # Salva o arquivo na pasta data/
        save_path = os.path.join(DATA_FOLDER, filename)
        file.stream.seek(0)
        content = file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        # Nome sem extensão
        file_name_no_ext = os.path.splitext(filename)[0]

        # Monta dicionário no formato esperado pela pipeline
        dados = {
            "path": save_path,
            "file_name": file_name_no_ext,
            "turnover_col": turnover_col
        }

        # Validação (corrigida)
        erros = validar_entrada(dados)
        if erros:
            logging.warning("Erros de validação encontrados: %s", erros)
            return jsonify({"erro": "Entrada inválida", "detalhes": erros}), 400

        logging.info("Iniciando execução da pipeline...")
        resultado = main(dados)
        logging.info("Pipeline concluída com sucesso.")

        return jsonify({"status": "ok", "resultado": resultado})

    except Exception as e:
        logging.exception("Erro interno durante execução da interface")
        return jsonify({"erro": "Erro interno", "mensagem": str(e)}), 500


@app.route('/executar', methods=['POST'])
def executar():
    try:
        entrada_json = request.get_json()
        erros = validar_entrada(entrada_json)

        if erros:
            logging.warning("Foram encontrados problemas na requisição: %s", erros)
            return jsonify({"erro": "Entrada inválida", "detalhes": erros}), 400

        resultado = main(entrada_json)

        logging.info("Processamento dos dados concluído.")

        return jsonify(resultado)
    except Exception as e:
        logging.exception("Erro interno durante processamento")
        return jsonify({'erro': 'Erro interno', 'mensagem': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=app.config.get("DEBUG", True))
