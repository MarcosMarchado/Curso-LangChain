from langchain.chains.llm import LLMChain
from langchain.chains.transform import TransformChain
from langchain.chains.sequential import SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# --- Chain 1: Gera a query de busca ---
llm = OpenAI(temperature=0.7)
prompt_busca = PromptTemplate(
    input_variables=["tema"],
    template="Escreva uma query de busca no Google sobre: {tema}. Retorne APENAS a query."
)
chain_busca = LLMChain(llm=llm, prompt=prompt_busca, output_key="query")

# --- Chain 2: Selenium faz a pesquisa e extrai resultados ---
def scrape_google(inputs: dict) -> dict:
    query = inputs["query"]
    
    # Configura o Selenium
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(f"https://www.google.com/search?q={query}")
    time.sleep(3)  # Aguarda carregar

    # Extrai os títulos e links dos resultados
    resultados = []
    for elemento in driver.find_elements(By.CSS_SELECTOR, "h3"):
        if elemento.text:
            resultados.append(elemento.text)
    
    driver.quit()
    return {"resultados": resultados[:3]}  # Retorna os 3 primeiros

chain_selenium = TransformChain(
    input_variables=["query"],
    output_variables=["resultados"],
    transform=scrape_google
)

# --- Chain 3: Resume os resultados com LLM ---
prompt_resumo = PromptTemplate(
    input_variables=["resultados"],
    template="Resuma os seguintes resultados de busca em 2 frases:\n{resultados}"
)
chain_resumo = LLMChain(llm=llm, prompt=prompt_resumo, output_key="resumo")

# --- Chain Final ---
chain_final = SequentialChain(
    chains=[chain_busca, chain_selenium, chain_resumo],
    input_variables=["tema"],
    output_variables=["resumo"],
    verbose=True
)

# --- Execução ---
resposta = chain_final.run(tema="inteligência artificial em 2024")
print(resposta)