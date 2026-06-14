# rag_core.py
import os
import re
import sys
import json
import unicodedata
from dotenv import load_dotenv, find_dotenv
from typing import Optional, Any, Dict, List

# Langchain imports
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain import tools

# Ensure your custom ConectaPGVector class is in PYTHONPATH or properly imported
# Ajuste este caminho se ConectaPGVector.py estiver em outro lugar
sys.path.append(r"C:\rpa\Python") 
from Classes.Postgres.Postgres.ConectaPGVector import ConectaPGVector

# Obtém o caminho do diretório onde o script está localizado
script_dir = os.path.dirname(os.path.abspath(__file__))
# Procura o .env a partir do diretório do script
dotenv_path = find_dotenv(os.path.join(script_dir, '.env'))
# Carrega o .env
load_dotenv(dotenv_path)
groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Adicionado: Token Secreto para a API
API_SECRET_TOKEN = os.getenv("API_SECRET_TOKEN") 
if API_SECRET_TOKEN is None:
    raise ValueError("API_SECRET_TOKEN não configurado nas variáveis de ambiente. Por favor, adicione-o ao seu arquivo .env.")

COLLECTION_NAME_PLANILHA = "isencoes_de_produtos"

# --- Componentes de IA (Serão Inicializados uma única vez na função initialize_rag_components) ---
# Usamos Optional para indicar que podem ser None antes da inicialização
# llm_model: Optional[ChatGroq] = None
llm_model: Optional[ChatOpenAI] = None
raw_pg_vector_store: Optional[Any] = None # Tipo Any para PGVector
agent_executor: Optional[AgentExecutor] = None

# --- Funções Auxiliares de Pré-processamento ---
def remover_miligramagem(principio_ativo: str) -> str:
    """Remove unidades de miligramagem e outras da string do princípio ativo."""
    return re.sub(r'\s?\d+(\.\d+)?\s*(mg/ml|mg|g|ml|mcg|kg|mcg|oz|ml|tablet|cap)', '', principio_ativo, flags=re.IGNORECASE).strip()


# --- Função para remover o ânion ---
def remover_anion(principio_ativo: str) -> str:
    """Remove unidades de miligramagem e outras da string do princípio ativo."""
    return re.sub(r'\s*(cloreto|sulfato|nitrato|fosfato|bicarbonato|acetato|tiossulfato|lactato|citrato|hidróxido|carbonato|borato|fluoreto|iodeto|brometo|peróxido|sulfeto|manganato|permanganato|cianeto|tartato|fosfonato|tetraborato|ácido sulfidrico|hidrogênico|oxalato|fenilalanato|ácido metanosulfônico|ácido acético|acetato de cálcio|etilenodiaminotetraacetato|dicromato|ácido clorosulfônico|bifluoreto|sulfito|metabisulfito|ácido fosfônico|sulfato de alumínio|ácido tartárico|ácido maleico|ácido fumárico|cloridrato|de)', '', principio_ativo, flags=re.IGNORECASE).strip()


def extrair_principios_ativos(principio_ativo_completo: str) -> List[str]:
    """Extrai e limpa múltiplos princípios ativos de uma string."""
    principios = principio_ativo_completo.split('+')
    principios_limpos = [remover_anion(remover_miligramagem(p.strip())) for p in principios]
    # principios_limpos = [(remover_miligramagem(p.strip())) for p in principios]
    print(principios_limpos)
    return principios_limpos


def busca_principio_ativo(principios_ativos: List[str], current_vector_store: Any) -> List[Any]:
    """Realiza busca por similaridade e filtra por correspondência exata do princípio ativo."""
    # O uso de um dicionário é um método robusto para deduplicar objetos mutáveis.
    # A chave será uma tupla da página e da fonte, que são hashable.
    all_retrieved_docs = {}
    for principio in principios_ativos:
        # retrieved_docs = current_vector_store.similarity_search(principio, k=20)
        retriever = current_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})

        retrieved_docs = retriever.invoke(principio)
        
        # filtered_docs = [
        #     doc for doc in retrieved_docs 
        #     if principio.lower() in doc.page_content.lower()
        # ]
        
        for doc in retrieved_docs:
            # Cria uma chave única e hashable para o dicionário
            doc_key = (doc.page_content, tuple(doc.metadata.items()))
            all_retrieved_docs[doc_key] = doc
    
    return list(all_retrieved_docs.values())


def buscar_por_ncm(ncm_usuario: str, current_vector_store: Any) -> List[Any]:
    """Realiza busca por similaridade de NCM e filtra resultados."""
    # results = current_vector_store.similarity_search(ncm_usuario, k=20)

    # 1. Obtém o retriever
    retriever = current_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
    # 2. Invoca o retriever para obter os documentos
    results = retriever.invoke(ncm_usuario)

    resultados_filtrados = []
    for doc in results:
        ncm_metadata_str = doc.metadata.get("ncm", "")
        ncm_list_in_doc = [n.strip() for n in ncm_metadata_str.split(',') if n.strip()]
        if ncm_usuario in ncm_list_in_doc:
            resultados_filtrados.append(doc)
    return resultados_filtrados


def normalize(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII").casefold()

def checar_correspondencia_completa(principios_produto, docs):
    for doc in docs:
        principio_doc = doc.metadata.get("principio_ativo", "").strip()
        if all(normalize(p) in normalize(principio_doc) for p in principios_produto):
            return doc
    return None


# --- Ferramenta e Lógica do Agente ---
def _obter_resposta_qa_chain(query: str, context: List[Any], llm_model_instance: ChatOpenAI) -> str:
    """Função interna para invocar a cadeia de QA."""
    # qa_system_prompt = '''
    # Você é um assistente especializado em isenção de ICMS.
    # Sua única e exclusiva tarefa é analisar o contexto fornecido para determinar se um produto é isento de ICMS e, se for, identificar o convênio ICMS vinculado.

    # **Contexto Fornecido:**
    # Cada entrada no contexto descreve uma isenção. O formato é: "Isenção: [Nome da Isenção]. Princípio Ativo: [Princípio Ativo]. Observação: [Observação]. NCMs relacionados: [Lista de NCMs]."
    # A parte "[Nome da Isenção]" contém o número do convênio (ex: Convênio 126/10).

    # **Instruções de Análise e Resposta (CRÍTICO - SIGA A ORDEM RIGOROSAMENTE):**
    # 1.  **PRIMEIRO - Busque apenas pelo Princípio Ativo:**
    #     * Examine a lista de documentos no contexto.
    #     * Para cada documento, verifique se o princípio ativo do produto (ex: OLAPARIBE) aparece no campo "Princípio Ativo" do documento.
    #     * **Se encontrar uma correspondência exata, o produto é isento.** Use a informação deste documento para a resposta final e PARE de analisar.
    #     * Se houver mais de um princípio ativo no produto, AMBOS devem estar presentes no mesmo documento para ser considerado isento.
    # 2.  **SEGUNDO - Busque pelo princíoio ativo + NCM (Somente se o Princípio Ativo NÃO foi encontrado):**
    #     * Se a etapa 1 falhou (o princípio ativo do produto não foi encontrado em nenhum documento), você deve prosseguir para a análise do princípio ativo do produto + NCM.
    #     * Examine a lista de documentos no contexto.
    #     * Para cada documento, verifique se o princípio ativo e o NCM do produto (ex: 30049069) estão na lista de "NCMs relacionados".
    #     * **Se encontrar um documento que satisfaça AMBAS as condições, o produto é isento.** Use a informação deste documento para a resposta final e PARE de analisar.
    # 3.  **TERCEIRO - Busque apenas pelo NCM:**
    #     * Se as etapas anteriores falharam, analise apenas o NCM do produto.
    #     * Para cada documento, verifique se o NCM do produto (ex: 30049069) está na lista de "NCMs relacionados". **E se o campo "Princípio Ativo" do documento e também do produto de busca está VAZIO. CONSIDERE ISSO COMO UMA REGRA IMPORTANTE**.
    #     * **Se encontrar um documento que satisfaça AMBAS as condições, o produto é isento.** Use a informação deste documento para a resposta final e PARE de analisar.
    # 4.  **Condição de Isenção:**
    #     * Se você encontrou uma isenção válida em qualquer um dos passos acima, o produto é isento.
    #     * Se nenhuma das condições acima for satisfeita, o produto NÃO é isento.
    # 5.  **Formato da Resposta (MUITO IMPORTANTE - Siga ESTE FORMATO EXATAMENTE):**
    #     * **Se o produto for isento:**
    #         * Responda: "Sim. [NÚMERO_DO_CONVÊNIO]".
    #         * O [NÚMERO_DO_CONVÊNIO] deve ser extraído do documento correspondente (ex: 126/10).
    #     * **Se o produto NÃO for isento:**
    #         * Responda: "Não".

    # **Exemplos de Resposta Esperada:**
    # * "Sim. Convênio ICMS 126/10"
    # * "Sim. Convênio ICMS 10/02"
    # * "Sim. Convênio ICMS 162/94 - 132/2021"
    # * "Não"

    # Me retorne apenas a resposta direta, sem mais informações ou explicações adicionais.

    # Contexto: {context}
    # '''
    qa_system_prompt = """
    Você é um assistente especializado em isenção de ICMS.
    Sua única e exclusiva tarefa é analisar o contexto fornecido para determinar se um produto é isento de ICMS e, se for, identificar o convênio ICMS vinculado.

    **Contexto Fornecido:**
    Cada entrada no contexto descreve uma isenção. O formato é: 
    "Isenção: [Nome da Isenção]. Princípio Ativo: [Princípio Ativo]. Observação: [Observação]. NCMs relacionados: [Lista de NCMs]."
    A parte "[Nome da Isenção]" contém o número do convênio (ex: Convênio 126/10).

    **Instruções de Análise e Resposta (SIGA RIGOROSAMENTE):**

    1. **PRIMEIRA REGRA – Apenas pelo Princípio Ativo:**
    * Examine a lista de documentos no contexto.
    * Para cada documento, verifique se o princípio ativo do produto aparece no campo "Princípio Ativo" do documento.
    * Se o produto tiver mais de um princípio ativo, TODOS devem estar presentes no mesmo documento para ser considerado isento.
    * Desconsidere a busca pelo NCM aqui, considere apenas o princípio ativo.
    * Se encontrar correspondência, o produto é isento. Use a informação deste documento para a resposta final e PARE de analisar.

    2. **SEGUNDA REGRA – Apenas pelo NCM (condição de vazio):**
    * Só utilize esta regra se o princípio ativo do produto for VAZIO **e** o princípio ativo do documento também for VAZIO.
    * Nesse caso, compare apenas o NCM do produto com os NCMs listados no documento.
    * Se houver correspondência, o produto é isento. Use a informação deste documento para a resposta final e PARE de analisar.

    3. **Condição Final de Isenção:**
    * Se encontrou uma isenção válida em qualquer uma das duas regras acima, o produto é isento.
    * Caso contrário, o produto NÃO é isento.

    4. **Formato da Resposta (OBRIGATÓRIO):**
    * Se o produto for isento:
        - Responda: "Sim. Convênio ICMS [NÚMERO_DO_CONVÊNIO]"
    * Se o produto NÃO for isento:
        - Responda: "Não"

    **Exemplos de Resposta Esperada:**
    * "Sim. Convênio ICMS 126/10"
    * "Sim. Convênio ICMS 10/02"
    * "Sim. Convênio ICMS 162/94 - 132/2021"
    * "Não"

    Contexto: {context}
    """
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
        
        # if principio_ativo:
        #     principios_ativos = extrair_principios_ativos(principio_ativo)
        #     retrieved_docs = busca_principio_ativo(principios_ativos, raw_pg_vector_store)
        
        # if not retrieved_docs and ncm:
        #     retrieved_docs = buscar_por_ncm(ncm, raw_pg_vector_store)

        # if principio_ativo:
        #     principios_ativos = extrair_principios_ativos(principio_ativo)
        #     retrieved_docs = busca_principio_ativo(principios_ativos, raw_pg_vector_store)
        # else:
        #     if ncm:
        #         retrieved_docs = buscar_por_ncm(ncm, raw_pg_vector_store)

        if principio_ativo:
            principios_ativos = extrair_principios_ativos(principio_ativo)
            retrieved_docs = busca_principio_ativo(principios_ativos, raw_pg_vector_store)
            
            matched_doc = checar_correspondencia_completa(principios_ativos, retrieved_docs)
            if matched_doc:
                conv = matched_doc.metadata.get("nome_isencao")
                return f"Sim. Convênio ICMS {conv}"
            else:
                # Aqui entra o LLM para analisar múltiplos princípios ativos
                if retrieved_docs:
                    query = f"Verifique se o produto com princípios ativos {principios_ativos} é isento de ICMS considerando os documentos fornecidos."
                    return _obter_resposta_qa_chain(query, retrieved_docs, llm_model)
                else:
                    if ncm:
                        retrieved_docs = buscar_por_ncm(ncm, raw_pg_vector_store)
        
        print(retrieved_docs)
        
        query = f"""
        Verifique se o produto é isento de ICMS. 
        Regras obrigatórias:
        1. Se o produto tiver princípio ativo, considere apenas o princípio ativo e ignore totalmente o NCM. 
        Se o princípio ativo bater com o documento, retorne o convênio.
        2. Se o produto NÃO tiver princípio ativo, compare apenas o NCM do produto com os NCMs do documento, 
        mas somente se o campo "Princípio Ativo" do documento também estiver vazio.
        3. Se não houver correspondência, responda "Não".

        Produto:
        Princípio ativo: {principio_ativo}
        NCM: {ncm}
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
    # llm_model = ChatGroq(
    #     model="meta-llama/llama-4-maverick-17b-128e-instruct",
    # )
    llm_model = ChatOpenAI(
        # model="gpt-4.1-mini",
        # model="gpt-5-mini",
        model="gpt-4o-mini",
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
    Você é um assistente que determina a isenção de ICMS de produtos e formata a resposta em JSON.

    **Sua TAREFA ÚNICA E FINAL é:**
    1. Usar a ferramenta 'get_icms_exemption' EXATAMENTE uma vez com o input da pergunta.
    2. Pegar a resposta da ferramenta (ex: 'Sim. 162/94 - 132/2021' ou 'Não').
    3. Converter ESSA resposta para o formato JSON final e fornecer APENAS ESSE JSON como sua FINAL ANSWER.

    **ATENÇÃO: A entrada para a ferramenta 'get_icms_exemption' deve ser uma string JSON válida, sem aspas extras.**
    **ATENÇÃO: A sua Final Answer DEVE ter apenas as chaves 'isento' e 'convenio'. Não use outras chaves.**
    **ATENÇÃO: Pense passo-a-passo e com calma.**

    Available tools:
    {tools}
    Ferramentas que você pode usar: {tool_names} 

    Use o seguinte formato de raciocínio ReAct:

    Question: A pergunta ou tarefa de entrada.
    Thought: Você deve sempre pensar no que fazer. Minha primeira ação DEVE ser chamar a ferramenta 'get_icms_exemption' com o input JSON.
    Action: A ação a ser tomada, deve ser uma de [{tool_names}].
    Action Input: # O JSON deve ser colocado sem aspas extras. Por exemplo: {{"codigo": "123", ...}}
    Observation: O resultado da ação.
    Thought: Eu obtive a resposta da ferramenta. Minha ÚNICA tarefa agora é formatar a resposta no JSON e dar a Final Answer.
    Final Answer: # O objeto JSON virá aqui. NADA MAIS.

    Começar!

    Question: {input}
    Thought: {agent_scratchpad}
    """
    agent_prompt = PromptTemplate.from_template(agent_template_string)
    # agent_prompt = agent_prompt.partial(tool_names=", ".join([tool.name for tool in tools_for_agent]))

    agent = create_react_agent(llm=llm_model, tools=tools_for_agent, prompt=agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools_for_agent, verbose=True, # Definir como False para produção
        handle_parsing_errors=True, max_iterations=15,
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
        
        parsed_response = {}
        try:
            parsed_response = json.loads(raw_output_string)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*?\}', raw_output_string, re.DOTALL)
            if json_match:
                json_str_candidate = json_match.group(0)
                try:
                    parsed_response = json.loads(json_str_candidate)
                except json.JSONDecodeError:
                    parsed_response = {"isento_icms": "Erro de Formato", "convenio": None}
            else:
                parsed_response = {"isento_icms": "Erro de Formato", "convenio": None}
        
        isento_status = parsed_response.get("isento_icms") or parsed_response.get("isento")
        
        if isento_status and isento_status.lower() == "não":
            final_response = {"isento_icms": "Não", "convenio": None}
        elif isento_status:
            final_response = {"isento_icms": "Sim", "convenio": parsed_response.get("convenio", parsed_response.get("convenio_icms"))}
        elif not isento_status:
            final_response = {"isento_icms": "Não", "convenio": None}
        else:
            final_response = {"isento_icms": "Erro Inesperado", "convenio": None}

        return final_response

    except Exception as e:
        print(f"Erro geral ao processar produto com o agente: {e}")
        return {"isento_icms": "Erro Inesperado", "convenio": None}
    

# Para testar rag_core.py individualmente (opcional):
if __name__ == "__main__":
    # Inicializa os componentes APENAS se este script for executado diretamente
    print("Executando rag_core.py diretamente para teste...")
    initialize_rag_components() # Chama a inicialização
    
    # test_product = {
    #     "codigo": "765951",
    #     "nome": "KOSELUGO10MG 60CAPS",
    #     "principio_ativo": "SULFATO DE SELUMETINIBE 10 MG",
    #     "ncm": "30049079"
    # }
    test_product = {
        "codigo": "708488",
        "nome": "SB REXONA LIMPEZA PROFUNDA 6UN",
        "principio_ativo": "",
        "ncm": "34011190"
    }
    result = process_product_with_rag(**test_product) # Não precisa passar agent_executor aqui
    print("\nResultado do teste direto (rag_core.py):")
    print(json.dumps(result, indent=2))