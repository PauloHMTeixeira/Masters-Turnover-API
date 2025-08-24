import requests
import json

# URL da sua API
url = "http://127.0.0.1:5000/processa"

# Exemplo de payload
payload = {
    "path": "data/IBM-HR-Employee-Attrition.csv",
    "file_name": "teste"
}

# Envia a requisição POST
response = requests.post(url, json=payload)

# Exibe a resposta
print("Status code:", response.status_code)
print("Resposta JSON:", response.json())