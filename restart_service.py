import subprocess
import time
import sys

sys.path.append(r"C:\rpa\Python")
from Classes.Hangouts.Hangouts.Hangouts import Hangouts

def restart_windows_service(service_name: str):
    """
    Para e reinicia um servico do Windows.
    Este script DEVE ser executado com privilégios de administrador.
    
    Args:
        service_name (str): O nome do servico do Windows a ser reiniciado.
    """
    print(f"Tentando parar o servico '{service_name}'...")
    
    # Comando para parar o servico
    stop_command = f"net stop \"{service_name}\""
    
    try:
        # Usamos subprocess.run para executar o comando
        # `shell=True` é necessario para comandos simples do shell.
        # `check=True` faz com que uma excecão seja levantada se o comando falhar.
        subprocess.run(stop_command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"Servico '{service_name}' parado com sucesso.")
    except subprocess.CalledProcessError as e:
        # Se o comando falhar, verificamos se foi porque o servico ja estava parado.
        if "não foi iniciado" in e.stderr or "not been started" in e.stderr:
            print(f"Aviso: O servico '{service_name}' ja estava parado.")
        else:
            print(f"Erro ao parar o servico '{service_name}'. Verifique as permissoes.")
            print(f"Saida de erro: {e.stderr}")
            sys.exit(1)
    
    # Adicionamos uma pequena pausa para garantir que o processo encerrou completamente
    time.sleep(2)
    
    print(f"Tentando iniciar o servico '{service_name}'...")
    
    # Comando para iniciar o servico
    start_command = f"net start \"{service_name}\""
    
    try:
        subprocess.run(start_command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"Servico '{service_name}' iniciado com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao iniciar o servico '{service_name}'. Verifique as permissoes.")
        print(f"Saida de erro: {e.stderr}")
        sys.exit(1)


def main():
    try:
        hangs = Hangouts(mensagem="Olá!\n*Reiniciando a API do treinamento da isenção de ICMS!*", url="https://chat.googleapis.com/v1/spaces/AAAArlZTvjY/messages?key='SECRET_REMOVED_BY_AI'&token=ttHMivztI9WS2LDEWN-uHPddSaJhP0OUjtN942NIxZE")
        hangs.retorna_google_chat()
        # Nome do servico que você quer reiniciar
        service_name_to_restart = "MyIcmsApiService"
        restart_windows_service(service_name_to_restart)
        print("\nProcesso de reinicio concluido.")
        hangs = Hangouts(mensagem="Olá!\n*API do treinamento da isenção de ICMS reiniciada!*", url="https://chat.googleapis.com/v1/spaces/AAAArlZTvjY/messages?key='SECRET_REMOVED_BY_AI'&token=ttHMivztI9WS2LDEWN-uHPddSaJhP0OUjtN942NIxZE")
        hangs.retorna_google_chat()

    except Exception as e:
        hangs = Hangouts(mensagem=f"Olá!\n*Há erro ao reiniciar o treinamento da isenção de ICMS*\n\nMensagem de erro: {e}", url="https://chat.googleapis.com/v1/spaces/AAAArlZTvjY/messages?key='SECRET_REMOVED_BY_AI'&token=ttHMivztI9WS2LDEWN-uHPddSaJhP0OUjtN942NIxZE")
        hangs.retorna_google_chat()



main()