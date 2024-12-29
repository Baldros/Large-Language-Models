# Libs para LLMs:
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import torch

# Libs auxiliares:
import os
import json

#  Templete:
template = '''
Reposta à questão abaixo.

Aqui está o contexto da conversa {context}

Questão: {question}

Resposta:
'''


# Instanciado Modelo:
model = OllamaLLM(model='wizard-vicuna-uncensored')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def conversation(context = ""):
    '''
        Função construida para gerenciar conversar com
    o modelo de LLM definido.

    # Entrada:
    str: Contexto, caso haja.

    # Saída:
    str: Contexto da conversa.
    '''
    from time import time

    # Mensagem inicial:
    print('Bem vindo ao ChatBot, digite "/bye" para sair.')

    while True:
        user_input = input('Prompt: ')
        if user_input.lower() == "/bye":
            break

        start = time()
        # Entrada:
        result = chain.invoke({"context":context, "question":user_input})
        final = time()

        # Resposta:
        print("Bot: ", result)
        print(f'{final-start}')

        # Salvando Contexto:
        context += f"\nUser:{user_input}\nAI: {result}"
    
    # Fim da Mensagem:
    print('Bye, volte quando quiser. :)')

    return context

def RemeberContext(json_direct=r'C:\Users\amori\Cods\Agents\contexts.json'):
    '''
        Função construida para relembrar o
    contexto de alguma conversa em específico.

    # Entradas:
    str: Diretório do json de conversas.

    # Saída:
    dic: Conversas já tidas.
    '''

    # lendo Arquivos:
    with open(json_direct,'r') as arq:
        conversas = json.load(arq)

    return conversas

def save_context(context, json_direct='contexts.json'):
    '''
        Função para armazenar e gerenciar armazenamento
    de conversas. O objetivo aqui é não perder as conversas
    já obtidas de modo que elas possam ser continuadas.
    '''
    if context == '' or context == '/bye':
        print('Contexto Vazio')
        return

    # Abrir e ler o arquivo JSON
    try:
        with open(json_direct, 'r', encoding='ISO-8859-1') as arquivo:
            conversas = json.load(arquivo)
    except FileNotFoundError:
        # Se o arquivo não existe, inicializa um dicionário vazio
        conversas = {}

    # Gerando um título:
    title_context = model.invoke(input=f'Gere um título para essa conversa:\n {context}')

    # Adicionando nova conversa na estrutura de conversas
    conversas[title_context] = context

    # Salvando o dicionário atualizado de volta em um arquivo JSON
    with open(json_direct, 'w', encoding='ISO-8859-1') as arquivo:
        json.dump(conversas, arquivo, ensure_ascii=False, indent=4)

    print('Conversa Salva.')

if __name__ == '__main__':
    perg = input('Nova Conversa?(y/n)\n')

    if perg.lower() == 'y' or perg.lower() == 'yes':
        chat = conversation()
        save_context(chat)

    else:
        conversas = RemeberContext()

        # Listando conversas:
        for i,key in enumerate(conversas.keys()):
            print(f'{i} -> {key}')

        j = int(input('insira o numero da conversa: '))

        try:
            chat = conversation(conversas[list(conversas.keys())[j]])

            # Salvando Conversa:
            conversas[list(conversas.keys())[j]] += chat

            # Atualizando json:
            with open(r'C:\Users\amori\Cods\Agents\contexts.json', 'w', encoding='ISO-8859-1') as arquivo:
                json.dump(conversas, arquivo, ensure_ascii=False, indent=4)

        except TypeError as e:
            print(e)

        
