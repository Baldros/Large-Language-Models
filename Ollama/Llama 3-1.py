# Importando Dependências:
from llama_index.llms.ollama import Ollama
from time import time

start = time()
# Instanciando Modelo:
llm = Ollama(model='llama3.1', request_timeout=120.0)

# Prompt
entrada = input('Prompt:\n')
resp = llm.complete(entrada)
final = time()

print(resp)
print(f'{final - start}s')