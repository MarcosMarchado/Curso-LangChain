import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from langchain.chains.transform import TransformChain
from langchain.chains.sequential import SequentialChain
from langchain.chains.llm import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

import datetime
import time
import os

# ========== (1) Chain 1: Extrai notícias com Selenium ==========
def scrape_news(inputs: dict) -> dict:
    query = inputs["tema"]
    
    # Configura o Selenium para buscar notícias no Google
    driver = webdriver.Chrome(service=webdriver.chrome.service.Service(ChromeDriverManager().install()))
    driver.get(f"https://www.google.com/search?q={query}&tbm=nws")  # Abre a aba "Notícias"
    time.sleep(3)
    
    # Pega os títulos dos resultados
    news_titles = []
    for title in driver.find_elements(By.CSS_SELECTOR, "div.nDgy9d"):  # Classe dos títulos de notícias
        if title.text:
            news_titles.append(title.text)
    
    driver.quit()
    return {"titles": news_titles[:3]}  # Retorna os 3 primeiros

chain_selenium = TransformChain(
    input_variables=["tema"],
    output_variables=["titles"],
    transform=scrape_news
)

# ========== (2) Chain 2: Classifica sentimentos com LLM ==========
llm = OpenAI(temperature=0.3)  # Modelo para análise de sentimentos

prompt_sentiment = PromptTemplate(
    input_variables=["title"],
    template="""
    Classifique o sentimento deste título de notícia como "POSITIVO", "NEGATIVO" ou "NEUTRO":
    
    Título: {title}
    Sentimento:"""
)

def classify_sentiments(inputs: dict) -> dict:
    titles = inputs["titles"]
    results = []
    for title in titles:
        sentiment = llm(prompt_sentiment.format(title=title))
        results.append({"title": title, "sentiment": sentiment.strip()})
    return {"news_data": results}

chain_sentiment = TransformChain(
    input_variables=["titles"],
    output_variables=["news_data"],
    transform=classify_sentiments
)

# ========== (3) Chain 3: Salva no Excel ==========
def save_to_excel(inputs: dict) -> dict:
    news_data = inputs["news_data"]
    df = pd.DataFrame(news_data)
    df["date"] = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Define o caminho completo para o arquivo
    caminho_pasta_pai = "automacoes"
    caminho_pasta = os.path.join(caminho_pasta_pai, "src")
    caminho_arquivo = os.path.join(caminho_pasta, "news_sentiment.xlsx")
    
    # Garante que a pasta existe (cria se não existir)
    os.makedirs(caminho_pasta, exist_ok=True)
    
    # Salva ou atualiza o Excel
    try:
        existing_df = pd.read_excel(caminho_arquivo)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        updated_df = df
    
    # Salva o arquivo
    updated_df.to_excel(caminho_arquivo, index=False)
    return {"status": "Dados salvos com sucesso!"}

chain_excel = TransformChain(
    input_variables=["news_data"],
    output_variables=["status"],
    transform=save_to_excel
)

# ========== EXECUÇÃO FINAL ==========
chain_final = SequentialChain(
    chains=[chain_selenium, chain_sentiment, chain_excel],
    input_variables=["tema"],
    output_variables=["status"],
    verbose=True
)

result = chain_final.run(tema="mundial de clubes")
print(result)