# Dependências:
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from time import time


# Instanciando modelo:
llm=OllamaLLM(model='llama3.1')

# Função:
def oscar(filme, ano, llm=llm):
    prompt = PromptTemplate(
        input_variables=['filme','ano'],
        template="Quantos oscars o filme {filme} ganhou em {ano}"
    )

    oscar_chain = prompt | llm
    start = time()
    resposta = oscar_chain.invoke({'filme':filme,'ano':ano})
    final = time()
    print(f'{round(final-start,2)}s')

    return resposta


if __name__=='__main__':
    resposta = oscar('Oppenheimer',2024)
    print(resposta)
