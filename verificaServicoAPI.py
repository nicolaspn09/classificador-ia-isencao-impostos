import sys
import subprocess

sys.path.append(r"C:\rpa\Python")
from Classes.Hangouts.Hangouts.Hangouts import Hangouts

def verificar_servico(nome_servico):
    # Comando para verificar o status do serviço
    comando = f'sc qc {nome_servico}'
    resultado = subprocess.run(comando, capture_output=True, text=True, shell=True)

    if resultado.returncode == 0:
        if "RUNNING" in resultado.stdout:
            print(f"O serviço {nome_servico} está em execução. Parando o serviço...")
            hangs = Hangouts(mensagem=f"O serviço {nome_servico} está em execução. Parando o serviço...", url="https://chat.googleapis.com/v1/spaces/AAAAG2VLaCI/messages?key='SECRET_REMOVED_BY_AI'&token=ipYvgtYxzRGjUfPmgf8VNACVbgeZzgYoRFcZnWekSfo")
            hangs.retorna_google_chat()
            parar_servico(nome_servico)
        else:
            hangs = Hangouts(mensagem=f"O serviço {nome_servico} já está parado.", url="https://chat.googleapis.com/v1/spaces/AAAAG2VLaCI/messages?key='SECRET_REMOVED_BY_AI'&token=ipYvgtYxzRGjUfPmgf8VNACVbgeZzgYoRFcZnWekSfo")
            hangs.retorna_google_chat()
            print(f"O serviço {nome_servico} já está parado.")
    else:
        hangs = Hangouts(mensagem=f"Erro ao consultar o serviço {nome_servico}: {resultado.stderr}", url="https://chat.googleapis.com/v1/spaces/AAAAG2VLaCI/messages?key='SECRET_REMOVED_BY_AI'&token=ipYvgtYxzRGjUfPmgf8VNACVbgeZzgYoRFcZnWekSfo")
        hangs.retorna_google_chat()
        print(f"Erro ao consultar o serviço {nome_servico}: {resultado.stderr}")

def parar_servico(nome_servico):
    # Comando para parar o serviço
    comando = f'net stop {nome_servico}'
    resultado = subprocess.run(comando, capture_output=True, text=True, shell=True)
    
    if resultado.returncode == 0:
        print(f"Serviço {nome_servico} parado com sucesso.")
        hangs = Hangouts(mensagem=f"Serviço {nome_servico} parado com sucesso.", url="https://chat.googleapis.com/v1/spaces/AAAAG2VLaCI/messages?key='SECRET_REMOVED_BY_AI'&token=ipYvgtYxzRGjUfPmgf8VNACVbgeZzgYoRFcZnWekSfo")
        hangs.retorna_google_chat()
    else:
        print(f"Erro ao parar o serviço {nome_servico}: {resultado.stderr}")

# Nome do serviço a ser verificado
nome_servico = "MyIcmsApiService"

# Verificar o status e garantir que o serviço está parado
verificar_servico(nome_servico)