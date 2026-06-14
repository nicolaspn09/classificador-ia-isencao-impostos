# cliente_rag.py
import requests
import json

# As credenciais da sua API de gatilho
API_GATILHO_URL = "'http://api.empresa.com.br'.202:8000/check-icms-exemption/"
TOKEN_DE_AUTENTICACAO = "3u2ry7935ybefrjkjth23974yr13" # Use o token exato do seu .env da API

# Dados do produto que você quer processar
product_data = {
    "codigo": "765951",
    "nome": "KOSELUGO10MG 60CAPS",
    "principio_ativo": "SULFATO DE SELUMETINIBE 10 MG",
    "ncm": "30049079"
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN_DE_AUTENTICACAO}"
}

print("Chamando API de Gatilho...")
try:
    response = requests.post(API_GATILHO_URL, headers=headers, data=json.dumps(product_data))
    response.raise_for_status()

    # A resposta da API de Gatilho será imediata, informando que a DAG foi disparada.
    gatilho_response = response.json()
    print("Resposta da API de Gatilho:")
    print(json.dumps(gatilho_response, indent=2))
    print("\nVerifique o log do Airflow para ver a execução da DAG...")

except requests.exceptions.RequestException as e:
    print(f"Erro ao chamar a API de Gatilho: {e}")