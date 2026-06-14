import sys

sys.path.append(r"C:\rpa\Python")
from Classes.GoogleSheets.GoogleSheets.GoogleSheets import GoogleSheets
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata 
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()



# Função para usar os dados da planilha no processo de RAG
def processar_planilha(id_planilha, range_dados, diretorio_json):
    try:
        os.remove(r"C:\rpa\Python\Modelo Isencao Produtos\db")
    except:
        pass

    # Cria a instância da classe GoogleSheets
    gs = GoogleSheets(id_planilha, range_dados, diretorio_json)
    
    # Obtém os dados da planilha
    dados_planilha = gs.solicita_tabela()

    # Processa os dados em "documentos"
    docs_to_add = []
    for row in dados_planilha:
        # Garante que as colunas existam antes de tentar acessá-las
        nome_isencao = row[0] if len(row) > 0 else ""
        # indice = row[1] if len(row) > 1 else "" # O índice pode ser desconsiderado conforme seu prompt original
        principio_ativo = row[2] if len(row) > 2 else ""
        observacao = row[3] if len(row) > 3 else ""
        
        # Coleta todos os NCMs (colunas 4 a 7) de forma mais robusta
        ncm_list = []
        for i in range(4, 8): # Itera das colunas 4 a 7
            if len(row) > i and row[i] and str(row[i]).strip(): # Verifica se a coluna existe e não está vazia
                ncm_list.append(str(row[i]).strip())
        
        # Constrói o conteúdo principal do documento dinamicamente
        doc_content_parts = []
        if nome_isencao:
            doc_content_parts.append(f"Isenção: {nome_isencao}.")
        if principio_ativo:
            doc_content_parts.append(f"Princípio Ativo: {principio_ativo}.")
        if observacao:
            doc_content_parts.append(f"Observação: {observacao}.")
        if ncm_list:
            doc_content_parts.append(f"NCMs relacionados: {', '.join(ncm_list)}.")
            
        doc_content = " ".join(doc_content_parts)
        
        # Prepara os metadados
        metadata = {
            "nome_isencao": nome_isencao,
            "principio_ativo": principio_ativo,
            "observacao": observacao,
            "ncm": ", ".join(ncm_list) # Transforma a lista de NCMs em uma string para os metadados
        }

        docs_to_add.append(Document(page_content=doc_content, metadata=metadata))

    # Configura o splitter para os chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Dividindo os dados em chunks
    # chunks = text_splitter.split_documents(documents=docs_to_add)
    
    # Cada linha completa é um chunk
    chunks = docs_to_add

    # Inicializa o embedding e o ChromaDB
    persist_directory = r"C:\rpa\Python\Modelo Isencao Produtos\db"
    embedding = HuggingFaceEmbeddings()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_name="planilha_data")

    # Adiciona os chunks ao ChromaDB
    try:
        vector_store.add_documents(documents=chunks)
        print("Chunks adicionados ao DB.")
    except Exception as e:
        print(f"Erro ao adicionar chunks ao ChromaDB: {e}")


if __name__ == "__main__":
    # Exemplo de como chamar a função
    id_planilha = "1rBlkAlRl0IJG0KJLAjpNsMAxw_G3YH03h8xW_Eskl6A"  # Substitua pelo ID da sua planilha
    range_dados = "A2:H"  # Substitua pelo range de dados que você deseja acessar
    diretorio_json = r"C:\rpa\Python\Modelo Isencao Produtos\token.json"  # Caminho para o seu arquivo de credenciais

    processar_planilha(id_planilha, range_dados, diretorio_json)
