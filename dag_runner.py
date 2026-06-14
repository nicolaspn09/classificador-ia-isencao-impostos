import sys
import json
import argparse
from typing import Optional

# Adiciona o caminho para as classes principais
# O caminho abaixo deve ser absoluto e o Python deve ter acesso a ele.
sys.path.append(r"C:\rpa\Python\Modelo Isencao Produtos")

# Importa a lógica de RAG do seu arquivo 'rag_core.py'
import rag_core

def main():
    """
    Função principal para executar a lógica de RAG.
    Recebe argumentos da linha de comando e processa um produto.
    """
    # Configurar o parser de argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Executa a lógica de RAG para verificar a isenção de ICMS.")
    parser.add_argument('--input_json', type=str, required=True, help='Dados do produto em formato JSON.')
    
    args = parser.parse_args()
    
    try:
        # Tenta parsear os dados JSON recebidos
        product_data = json.loads(args.input_json)
        
        # Chama a função de processamento de RAG com os dados
        result = rag_core.process_product_with_rag(
            codigo=product_data.get('codigo'),
            nome=product_data.get('nome'),
            principio_ativo=product_data.get('principio_ativo'),
            ncm=product_data.get('ncm')
        )
        
        # Imprime o resultado JSON para stdout
        # O SSHOperator vai capturar esta saída
        print(json.dumps(result))
        
    except json.JSONDecodeError:
        print(json.dumps({"status_isencao": "API_ERROR", "detalhe": "Erro ao parsear o JSON de entrada."}))
    except Exception as e:
        print(json.dumps({"status_isencao": "API_ERROR", "detalhe": f"Erro na execução do script: {e}"}))
    
if __name__ == '__main__':
    # A inicialização dos componentes RAG deve acontecer aqui,
    # dentro do script que será executado
    try:
        rag_core.initialize_rag_components()
        main()
    except Exception as e:
        print(json.dumps({"status_isencao": "API_ERROR", "detalhe": f"Falha na inicialização dos componentes: {e}"}))