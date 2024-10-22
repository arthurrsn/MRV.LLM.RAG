from IPython.display import display
import google.generativeai as genai
import pandas as pd
import numpy as np
import textwrap

# Configuração da API KEY
genai.configure(api_key='AIzaSyCz-icbgWmICzz1LP5Ah4dB0tIxCkdWzKQ') # Chave API
model_embed, model_generate = 'models/embedding-001', genai.GenerativeModel('gemini-1.5-pro-latest') # Modelos selecionado

# Configurando DF
df = pd.read_csv('dados.csv', delimiter=',') # definindo que os documentos passados vao ser um data frame

def criar_embed():
    def embed_fn(title, text):
        """
        descrição: Criar embeddings para o texto e o titulo passado, como estamos falando de uma base de dados com varios arquivos, então vamos adicionar essa função a um apply para aplicar a todos os itens selecionados

        params: title = Titulo do documento
                text = conteudo completo
        """
        return genai.embed_content(model=model_embed,
                                content=text,
                                task_type="retrieval_document",
                                title=title)["embedding"]

    # Criando uma copia de df para o data frame
    df_old = df.copy()

    # para cada linha ele irá executar a função embed_fn puxando o titulo e o text se o campo embeddings estiver vazio
    df['Embeddings'] = df.apply(
        lambda row: embed_fn(row['Title'], row['Text'])
        if pd.isna(row['Embeddings']) else row['Embeddings'],
        axis=1
    )

    # Salvando os dados no Arquivo CSV sem atribuir um novo index SE tiver alguma alteração
    if not df_old.equals(df):
        print('Salvo')
        df.to_csv('dados.csv', index=False)
    else:
        print('Não salvo')

criar_embed()

# !!! ULTIMA ATUALIZAÇÃO: !!!
# Até aqui o programa está conseguindo transformar nossos documentos em embeddings. Próximo passo é implementar o sistema de query
