import json
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials # Importar para autenticação
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Importar as funções e o módulo rag_core
import rag_core # O nome do seu arquivo de módulo refatorado

# --- Inicialização da Aplicação FastAPI ---
app = FastAPI(
    title="API de Verificação de Isenção de ICMS",
    description="API que utiliza um Agente de IA para verificar a isenção de ICMS de produtos com base em Princípio Ativo e NCM.",
    version="1.0.0"
)

# Esquema de segurança para Bearer Token
oauth2_scheme = HTTPBearer()

# Modelo Pydantic para os dados de entrada da requisição
class ProductData(BaseModel):
    codigo: Optional[str] = None
    nome: Optional[str] = None
    principio_ativo: Optional[str] = None
    ncm: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """
    Função que será executada uma única vez quando a API for iniciada.
    Aqui inicializamos todos os componentes RAG pesados através de rag_core.
    """
    print("Iniciando a API e inicializando componentes RAG...")
    try:
        rag_core.initialize_rag_components() # Chama a função de inicialização do rag_core
        print("Componentes RAG inicializados e prontos para uso.")
    except Exception as e:
        print(f"ERRO CRÍTICO NA INICIALIZAÇÃO DOS COMPONENTES RAG: {e}")
        raise RuntimeError("Falha na inicialização dos componentes RAG. A API não pode iniciar.") from e

# Função de dependência para verificar o token
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)):
    """
    Verifica o token de autorização fornecido na requisição.
    """
    # rag_core.API_SECRET_TOKEN é a variável global carregada do .env
    if credentials.credentials != rag_core.API_SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou não fornecido.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True # Retorna True se o token for válido

@app.get("/")
async def root():
    return {"message": "API de Verificação de Isenção de ICMS está online. Acesse /docs para a documentação."}

# Endpoint da API, agora protegido pelo token
@app.post("/check-icms-exemption/", response_model=Dict[str, Any])
async def check_icms_exemption_endpoint(
    product: ProductData,
    token_valid: bool = Depends(verify_token) # Adiciona a dependência do token
):
    """
    Verifica a isenção de ICMS para um produto fornecido e retorna o status
    e o convênio vinculado em formato JSON. Requer autenticação Bearer Token.
    """
    # A lógica principal é chamada apenas se o token for válido
    print(f"Requisição recebida para o produto: Código={product.codigo}, PA={product.principio_ativo}")
    try:
        # Chama a função de processamento de RAG do rag_core
        result_json = rag_core.process_product_with_rag(
            codigo=product.codigo,
            nome=product.nome,
            principio_ativo=product.principio_ativo,
            ncm=product.ncm
        )
        print(f"Resultado final para {product.codigo}: {json.dumps(result_json)}")
        return result_json
    except Exception as e:
        print(f"Erro ao processar requisição para {product.codigo}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar a requisição: {e}"
        )

# Para rodar esta API:
# 1. Salve o código acima como 'api_server.py' e 'rag_core.py' no mesmo diretório.
# 2. **Crie um arquivo .env** no diretório raiz do seu projeto com:
#    API_SECRET_TOKEN = 'REMOVED_FOR_GITHUB'
#    GROQ_API_KEY = 'REMOVED_FOR_GITHUB'
#    PG_HOST = 'REMOVED'
#    ... (outras credenciais PG)
# 3. Abra seu terminal no diretório do arquivo e ative seu ambiente virtual.
# 4. Execute: uvicorn api_server:app --reload --port 8000
# 5. Acesse a documentação interativa em http://127.0.0.1:8000/docs