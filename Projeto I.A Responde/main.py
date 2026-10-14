# pip install python-dotenv langchain langchain-openai langhchain-community langchain-chroma chroma-db openai pypdf

from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
CAMINHO_DB = "db"


prompt_template = """
Responda a pergunta do usuário:
{pergunta}
 
com base nessas informações:
{base_conhecimento}

Se você não encontrar a resposta da pergunta do usuário baseado nessas informações, responda não sei te dizer.
"""

pergunta = input("Escreva a sua pergunta: ")

#carregar do db com minhas informaçoes

funcao_embedding = OpenAIEmbeddings()
db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

#comparar a pergunta do usuário (embedding) com o banco de dados

resultados = db.similarity_search_with_relevance_scores(pergunta)
print(resultados)
print(len(resultados))