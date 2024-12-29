from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import conversational_retrieval
from vectordb import url_to_retriver
from LLM_Langchain import llm
from time import time

prompt = ChatPromptTemplate.from_template("""
Resposnda a pergunta com base apenas do contexto     
{context}

Pergunta: {input}""")

# Estruturando Rag:
document_chain = create_stuff_documents_chain(llm,prompt)
retriver = url_to_retriver('https://pt.wikipedia.org/wiki/Oppenheimer_(filme)')
retriver_chain = conversational_retrieval(retriver, document_chain)

#input = input('Insira a pergunta:\n')

# Estruturando Resposta:
start = time()
resposta = retriver_chain.invoke({"input":'Quantos oscars o filme Oppenheimer ganhou em 2024'})
final = time()
print(f'{round(final-start,2)}s')

# Resposta:
print(resposta)