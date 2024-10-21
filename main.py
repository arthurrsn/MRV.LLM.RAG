from IPython.display import display
import google.generativeai as genai
import pandas as pd
import numpy as np
import textwrap

# Configuração da API KEY
genai.configure(api_key='AIzaSyCz-icbgWmICzz1LP5Ah4dB0tIxCkdWzKQ') # Chave API
model_embed, model_generate = 'models/embedding-001', genai.GenerativeModel('gemini-1.5-pro-latest') # Modelos selecionado


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


#Documentos
DOCUMENT1 = {
    "title": "Operating the Climate Control System",
    "content": "Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."}
DOCUMENT2 = {
    "title": "Touchscreen",
    "content": "Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the \"Navigation\" icon to get directions to your destination or touch the \"Music\" icon to play your favorite songs."}
DOCUMENT3 = {
    "title": "Shifting Gears",
    "content": "Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."}

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

df = pd.DataFrame(documents) # definindo que os documentos passados vao ser um data frame 
df.columns = ['Title', 'Text'] # definindo que esses arquivos vao ficar separados por titulo e texto

df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1) # para cada linha ele irá executar a função embed_fn puxando o titulo e o text
display(df)

# !!! ULTIMA ATUALIZAÇÃO: !!!
# Até aqui o programa está conseguindo transformar nossos documentos em embeddings. Próximo passo é implementar o sistema de query
