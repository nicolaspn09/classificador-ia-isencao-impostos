# rag_core.py
import os
import re
import sys
import json
from dotenv import load_dotenv
from typing import Optional, Any, Dict, List

# Langchain imports
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain import tools

# Ensure your custom ConectaPGVector class is in PYTHONPATH or properly imported
# Ajuste este caminho se ConectaPGVector.py estiver em outro lugar
sys.path.append(r"C:\rpa\Python") 
from Classes.Postgres.Postgres.ConectaPGVector import ConectaPGVector

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Adicionado: Token Secreto para a API
API_SECRET_TOKEN = os.getenv("API_SECRET_TOKEN") 
if API_SECRET_TOKEN is None:
    raise ValueError("API_SECRET_TOKEN não configurado nas variáveis de ambiente. Por favor, adicione-o ao seu arquivo .env.")

COLLECTION_NAME_PLANILHA = "isencoes_de_produtos"

# --- Componentes de IA (Serão Inicializados uma única vez na função initialize_rag_components) ---
# Usamos Optional para indicar que podem ser None antes da inicialização
llm_model: Optional[ChatGroq] = None
raw_pg_vector_store: Optional[Any] = None # Tipo Any para PGVector
agent_executor: Optional[AgentExecutor] = None

# --- Funções Auxiliares de Pré-processamento ---
def remover_miligramagem(principio_ativo: str) -> str:
    """Remove unidades de miligramagem e outras da string do princípio ativo."""
    return re.sub(r'\s?\d+(\.\d+)?\s*(mg|g|ml|mcg|kg|mcg|oz|ml|tablet|cap)', '', principio_ativo, flags=re.IGNORECASE).strip()


def extrair_principios_ativos(principio_ativo_completo: str) -> List[str]:
    """Extrai e limpa múltiplos princípios ativos de uma string."""
    principios = principio_ativo_completo.split('+')
    principios_limpos = [remover_miligramagem(p.strip()) for p in principios]
    return principios_limpos


# --- Funções de Busca no Vector Store ---
def busca_principio_ativo(principios_ativos: List[str], current_vector_store: Any) -> List[Any]:
    """Realiza busca por similaridade de princípios ativos no vector store."""
    results = []
    for principio in principios_ativos:
        results.extend(current_vector_store.similarity_search(principio, k=20))
    return results


def buscar_por_ncm(ncm_usuario: str, current_vector_store: Any) -> List[Any]:
    """Realiza busca por similaridade de NCM e filtra resultados."""
    results = current_vector_store.similarity_search(ncm_usuario, k=20)
    resultados_filtrados = []
    for doc in results:
        ncm_metadata_str = doc.metadata.get("ncm", "")
        ncm_list_in_doc = [n.strip() for n in ncm_metadata_str.split(',') if n.strip()]
        if ncm_usuario in ncm_list_in_doc:
            resultados_filtrados.append(doc)
    return resultados_filtrados


# --- Ferramenta e Lógica do Agente ---
def _obter_resposta_qa_chain(query: str, context: List[Any], llm_model_instance: ChatGroq) -> str:
    """Função interna para invocar a cadeia de QA."""
    qa_system_prompt = '''
    Você é um assistente especializado em isenção de ICMS.
    Sua única e exclusiva tarefa é analisar o contexto fornecido para determinar se um produto é isento de ICMS e, se for, identificar o convênio ICMS vinculado.

    **Contexto Fornecido:**
    Cada entrada no contexto descreve uma isenção. O formato é: "Isenção: [Nome da Isenção]. Princípio Ativo: [Princípio Ativo]. Observação: [Observação]. NCMs relacionados: [Lista de NCMs]."
    A parte "[Nome da Isenção]" contém o número do convênio (ex: Convênio 126/10).

    **Instruções de Análise e Resposta (CRÍTICO - SIGA A ORDEM RIGOROSAMENTE):**
    1.  **PRIMEIRO - Busque pelo Princípio Ativo:**
        * Examine a pergunta do usuário. Se o produto tiver um "Princípio Ativo" (ex: OLAPARIBE 150 MG), ignore a miligramagem (ou qualquer outra quantidade) e considere APENAS o nome do princípio ativo (ex: OLAPARIBE).
        * Procure por este nome do princípio ativo no campo "Princípio Ativo" dos documentos do contexto.
        * **Se encontrar uma correspondência válida pelo Princípio Ativo, pare de procurar e use esta informação.**
    2.  **SEGUNDO - Busque pelo NCM (Somente se o Princípio Ativo NÃO foi encontrado):**
        * Se você NÃO encontrou uma correspondência pelo Princípio Ativo (ou se o Princípio Ativo não foi fornecido na pergunta), então, e somente então, verifique se o NCM do produto (ex: 30049069) está presente na lista de "NCMs relacionados" em qualquer documento do contexto.
        * **Se encontrar uma correspondência válida pelo NCM, use esta informação.**
    3.  **Condição de Isenção:** Se, e somente se, uma correspondência foi encontrada por qualquer um dos métodos acima (Princípio Ativo ou NCM), o produto é considerado isento de ICMS.
    4.  **Formato da Resposta (MUITO IMPORTANTE - Siga ESTE FORMATO EXATAMENTE):**
        * **Se o produto for isento:**
            * Responda: "Sim. Convênio ICMS [NÚMERO_DO_CONVÊNIO]".
            * O [NÚMERO_DO_CONVÊNIO] deve ser extraído da parte "Isenção: Convênio [NÚMERO_DO_CONVÊNIO]" do documento correspondente encontrado. Por exemplo, se o contexto diz "Isenção: Convênio 126/10", você deve usar "126/10".
            * Certifique-se de adicionar "ICMS" entre "Convênio" e o número.
        * **Se o produto NÃO for isento (se nenhuma correspondência válida foi encontrada pelo Princípio Ativo nem pelo NCM):**
            * Responda: "Não".

    **Exemplos de Resposta Esperada:**
    * "Sim. Convênio ICMS 126/10"
    * "Sim. Convênio ICMS 10/02"
    * "Sim. Convênio ICMS 162/94 - 132/2021"
    * "Não"

    Me retorne apenas a resposta direta, sem mais informações ou explicações adicionais.

    Contexto: {context}
    '''
    qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), ("human", "{input}"),])
    qa_chain = create_stuff_documents_chain(llm=llm_model_instance, prompt=qa_prompt,)
    return qa_chain.invoke({"input": query, "context": context})


def _obter_resposta_base_tool_func(input_str: str) -> str:
    """
    Função que atua como ferramenta do Agente.
    Converte dados JSON de entrada, busca no vector store e invoca a QA Chain.
    Utiliza as variáveis globais `raw_pg_vector_store` e `llm_model`.
    """
    if raw_pg_vector_store is None or llm_model is None:
        raise RuntimeError("Componentes RAG não inicializados. Chame initialize_rag_components primeiro.")

    try:
        product_data = json.loads(input_str)
        codigo = product_data.get("codigo")
        nome = product_data.get("nome")
        principio_ativo = product_data.get("principio_ativo")
        ncm = product_data.get("ncm")

        retrieved_docs = []
        
        if principio_ativo:
            principios_ativos = extrair_principios_ativos(principio_ativo)
            retrieved_docs = busca_principio_ativo(principios_ativos, raw_pg_vector_store)
        
        if not retrieved_docs and ncm:
            retrieved_docs = buscar_por_ncm(ncm, raw_pg_vector_store)
        
        if not retrieved_docs:
            return "Não" 
        
        query = f"""
        Verifique o contexto se o produto é isento de ICMS e se sim, me retorne o convênio vinculado. Analise primeiro pelo NCM, caso não encontre, analise pelo princípio ativo (o princípio ativo será destacado no produto que vou listar pra você). Me retorne apenas 'Sim' ou 'Não' e o convênio, sem mais informações.
        Pode ser que o produto não tenha princípio ativo ou NCM, nesse caso, verifique apenas a informação que está presente no produto.
        Se houver mais de um convênio, retorne todos os convênios separados por vírgula.

        Produto:
        Código: {codigo}
        Princípio ativo: {principio_ativo}
        NCM: {ncm}
        Nome: {nome}
        """
        
        response = _obter_resposta_qa_chain(query, retrieved_docs, llm_model)
        return response
    
    except json.JSONDecodeError:
        return "Erro: O input para a ferramenta obter_resposta_base_tool deve ser um JSON válido."
    except Exception as e:
        return f"Erro inesperado na ferramenta obter_resposta_base_tool: {e}"
    

def initialize_rag_components():
    """
    Inicializa todos os componentes pesados de RAG (LLM, PGVector, Agente).
    Esta função deve ser chamada apenas uma vez na inicialização da aplicação.
    """
    global llm_model, raw_pg_vector_store, agent_executor # Para modificar as variáveis globais

    print("Inicializando ConectaPGVector...")
    pg_vector_connector = ConectaPGVector()

    print(f"Obtendo instância direta do PGVector para a coleção: '{COLLECTION_NAME_PLANILHA}'...")
    raw_pg_vector_store = pg_vector_connector._get_vector_store(COLLECTION_NAME_PLANILHA)
    if raw_pg_vector_store is None:
        raise RuntimeError(f"Não foi possível obter a instância do PGVector para a coleção '{COLLECTION_NAME_PLANILHA}'. Verifique a conexão e o nome da coleção.")
    print("Instância do PGVector obtida com sucesso.")

    print("Inicializando LLM (Groq)...")
    llm_model = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
    )
    print("LLM inicializado.")

    # Criar a ferramenta a partir da função
    get_icms_exemption_tool = tools.Tool(
        name="get_icms_exemption",
        func=_obter_resposta_base_tool_func,
        description="Útil para determinar se um produto é isento de ICMS e qual convênio está vinculado. O input deve ser uma string JSON com 'codigo', 'nome', 'principio_ativo', 'ncm'."
    )
    tools_for_agent = [get_icms_exemption_tool]

    # Definir o Prompt para o Agente
    agent_template_string = """
    Você é um assistente especializado em analisar informações de produtos para determinar a isenção de ICMS e formatar a resposta.

    **Sua única tarefa é usar a ferramenta 'get_icms_exemption' EXATAMENTE uma vez para cada input.**
    Depois de obter a resposta da ferramenta, você deve formatá-la estritamente no formato JSON especificado abaixo e fornecer a FINAL ANSWER.
    NÃO chame a ferramenta novamente se você já tiver uma resposta dela.

    Available tools:
    {tools}
    Ferramentas que você pode usar: {tool_names} 

    Use o seguinte formato de raciocínio ReAct:
    Question: A pergunta ou tarefa de entrada que você deve responder.
    Thought: Você deve sempre pensar no que fazer. Se você já tem a resposta da ferramenta, sua próxima Thought deve ser sobre como formatar a FINAL ANSWER.
    Action: A ação a ser tomada, deve ser uma de [{tool_names}].
    Action Input: A entrada para a ação.
    Observation: O resultado da ação.
    ... (este Thought/Action/Action Input/Observation pode se repetir SOMENTE se a informação NÃO for suficiente após a primeira Observação)
    Thought: Eu agora sei a resposta final, e preciso formatá-la no JSON.
    Final Answer: a resposta final formatada estritamente no JSON.

    **Formato da Resposta JSON:**
    {{
    "status_isencao": "Sim" ou "Não",
    "convenio_icms": "NÚMERO_DO_CONVÊNIO" ou null (se não for isento)
    }}

    Exemplos de Saída Final (APENAS o JSON):
    {{"status_isencao": "Sim", "convenio_icms": "126/10"}}
    {{"status_isencao": "Não", "convenio_icms": null}}
    {{"status_isencao": "Sim", "convenio_icms": "162/94 - 132/2021"}}

    Começar!

    Question: {input}
    Thought: {agent_scratchpad}
    """
    agent_prompt = PromptTemplate.from_template(agent_template_string)
    agent_prompt = agent_prompt.partial(tool_names=", ".join([tool.name for tool in tools_for_agent]))

    agent = create_react_agent(llm=llm_model, tools=tools_for_agent, prompt=agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools_for_agent, verbose=False, # Definir como False para produção
        handle_parsing_errors=True, max_iterations=5, 
    )
    
    print("Componentes RAG inicializados com sucesso!")
    return agent_executor # Retorna o executor do agente já inicializado


def process_product_with_rag(
    codigo: Optional[str],
    nome: Optional[str],
    principio_ativo: Optional[str],
    ncm: Optional[str]
    ) -> Dict[str, Any]:
    """
    Processa os dados de um produto usando o Agente de RAG e retorna o resultado formatado.
    Esta função espera que o `agent_executor` já tenha sido inicializado globalmente.
    """
    global agent_executor # Garante que estamos usando a variável global
    if agent_executor is None:
        # Se a inicialização não foi feita (ex: em um script que não é a API, chamar initialize_rag_components())
        # Ou, para a API, isto é um erro crítico.
        raise RuntimeError("Agent Executor não inicializado. Chame initialize_rag_components() primeiro.")

    agent_input_data = {
        "codigo": codigo,
        "nome": nome,
        "principio_ativo": principio_ativo,
        "ncm": ncm
    }
    input_for_agent = json.dumps(agent_input_data)

    try:
        agent_response = agent_executor.invoke({"input": input_for_agent})
        raw_output_string = agent_response.get('output', '{"error": "Agent output missing"}')
        
        # --- Pós-processamento Robusto de JSON ---
        parsed_response = {} # Dicionário para armazenar o JSON parseado
        
        try:
            # 1. Tentar carregar como JSON diretamente (ideal se JSON Mode funcionar bem)
            parsed_response = json.loads(raw_output_string)

        except json.JSONDecodeError:
            # 2. Se falhar, tentar extrair o primeiro objeto JSON usando regex
            json_match = re.search(r'\{.*?\}', raw_output_string, re.DOTALL)
            
            if json_match:
                json_str_candidate = json_match.group(0) # Pega a string que parece JSON
                try:
                    parsed_response = json.loads(json_str_candidate)
                except json.JSONDecodeError:
                    parsed_response = {"status_isencao": "Erro de Formato", "convenio_icms": None, "detalhe": f"JSON inválido após regex: {json_str_candidate[:50]}..."}
            else:
                parsed_response = {"status_isencao": "Erro de Formato", "convenio_icms": None, "detalhe": f"Sem JSON na saída do agente: {raw_output_string[:50]}..."}
                
        # --- Fallback para garantir formato consistente (se parsed_response não tiver campos essenciais) ---
        if "status_isencao" not in parsed_response: 
             if "Sim. Convênio ICMS" in raw_output_string:
                 convenio_match = re.search(r'Convênio ICMS\s*([0-9/.\s-]+)', raw_output_string)
                 convenio = convenio_match.group(1).strip() if convenio_match else "Desconhecido"
                 parsed_response = {"status_isencao": "Sim", "convenio_icms": convenio}
             elif "Não" in raw_output_string and "Sim" not in raw_output_string:
                 parsed_response = {"status_isencao": "Não", "convenio_icms": None}
             else:
                 if "detalhe" not in parsed_response: 
                     parsed_response["status_isencao"] = "Erro Inesperado"
                     parsed_response["convenio_icms"] = None
                     parsed_response["detalhe"] = raw_output_string[:200] + "..."

        return parsed_response

    except Exception as e:
        print(f"Erro geral ao processar produto com o agente: {e}")
        return {"status_isencao": "Erro Inesperado", "convenio_icms": None, "detalhe": str(e)}
    

# Para testar rag_core.py individualmente (opcional):
if __name__ == "__main__":
    # Inicializa os componentes APENAS se este script for executado diretamente
    print("Executando rag_core.py diretamente para teste...")
    initialize_rag_components() # Chama a inicialização
    
    test_product = {
        "codigo": "766216",
        "nome": "OJJARA200MG30CPR",
        "principio_ativo": "DICLORIDRATO DE MOMELOTINIBE MONOIDRATADO 200 MG",
        "ncm": "30049079"
    }
    result = process_product_with_rag(**test_product) # Não precisa passar agent_executor aqui
    print("\nResultado do teste direto (rag_core.py):")
    print(json.dumps(result, indent=2))