# Importando Dependencias:
from langchain_text_splitters import RecursiveCharacterTextSplitter # Separa em pequenos Chunks
from langchain_community.document_loaders import WebBaseLoader # Vai scrapping de páginas da internet
from langchain_community.vectorstores import FAISS # Armazena num vector db
from langchain_ollama import OllamaEmbeddings # Tranforma em embeddings


def url_to_retriver(url):
    '''
        Função que gera o vectordb
    para a construção do RAG.
    '''
    # Fazendo scrapping da página:
    loader=WebBaseLoader(url)
    docs = loader.load()

    # Instanciando Embeddings:
    embeddings = OllamaEmbeddings(model='llama3.1')

    # Gerando Chunks:
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)

    # Gerando os embeddings:
    vector = FAISS.from_documents(documents,embeddings)

    # Recuperador:
    retriver = vector.as_retriever()

    return retriver