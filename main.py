# Importa bibliotecas necessárias
from IPython.display import display
import google.generativeai as genai
import pandas as pd
import numpy as np
import textwrap
import pyodbc

# Configuração da API do Google
genai.configure(api_key='AIzaSyCz-icbgWmICzz1LP5Ah4dB0tIxCkdWzKQ')  # Chave da API
model = 'models/embedding-001'  # Modelo para embeddings

# Configuração da conexão com o SQL Server
dados_conexao = (
    "Driver={SQL Server};"
    "Server=DESKTOP-BB8LNNA;"
    "Database=INFOSYSTEM;"
)

conexao = pyodbc.connect(dados_conexao)  # Conecta ao banco de dados
query = 'SELECT * FROM MRV_DadosProducao'  # Consulta SQL
df = pd.read_sql(query, conexao)  # Lê os dados da consulta em um DataFrame

# Função para gerar embeddings com base no título e conteúdo
def embed_fn(title, text):
    return genai.embed_content(model=model, content=text, task_type="retrieval_document", title=title)["embedding"]

# Aplica a função para criar a coluna de embeddings
df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Content']), axis=1)

while True:
    # Solicita consulta do usuário
    query = str(input('Input (ou digite "sair" para encerrar): '))
    
    # Condição para sair do loop
    if query.lower() == "sair":
        print("Chatbot encerrado.")
        break
    request = genai.embed_content(model=model, content=query, task_type="retrieval_query")

    # Função para encontrar a passagem mais relevante
    def find_best_passage(query, dataframe):
        query_embedding = genai.embed_content(model=model, content=query, task_type="retrieval_query")
        dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])  # Cálculo de similaridade
        idx = np.argmax(dot_products)  # Índice da passagem mais relevante
        return dataframe.iloc[idx]['Content']  # Retorna o conteúdo correspondente

    # Obtém a passagem relevante
    passage = find_best_passage(query, df)

    # Função para criar um prompt para o modelo
    def make_prompt(query, relevant_passage):
        escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")  # Escapa caracteres especiais
        prompt = textwrap.dedent(f"""Você é um bot útil e informativo que responde a perguntas usando texto da passagem de referência incluída abaixo. \
    Certifique-se de responder em uma frase completa, sendo abrangente, incluindo todas as informações de contexto relevantes. \
    No entanto, você está falando com um público não técnico, então certifique-se de quebrar conceitos complicados e \
    adotar um tom amigável e coloquial. \
    Se a passagem for irrelevante para a resposta, apenas finalize a conversa dizendo que não tem informações relevantes sobre o assunto.
    QUESTION: '{query}'
    PASSAGE: '{escaped}'
    
        ANSWER:
    """)  # Formata o prompt

        return prompt  # Retorna o prompt formatado

    prompt = make_prompt(query, passage)  # Cria o prompt

    # Inicializa o modelo de geração de texto
    model = genai.GenerativeModel('gemini-1.5-pro-latest')  # Modelo de geração
    answer = model.generate_content(prompt)  # Gera a resposta
    print(answer.text)  # Imprime a resposta gerada
