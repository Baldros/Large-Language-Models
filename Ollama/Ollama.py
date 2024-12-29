# Importando Dependencias:
import ollama
from time import time

start = time()
# Instanciando Modelo:
resposta = ollama.chat(model='llama3.1', messages=[
    {
        'role':'User',
        'content':'Fiz um código para um modelo de LLM usando o Ollama e o LangChain. Só que ele me devolve a respota de uma única vez... Eu consigo mudar isso e ver ele respondendo palavra por palavra? Tipo como o chatgpt responde?',
    }

])
final = time()
print(resposta['message']['content'])
print(f'{final - start}s')
