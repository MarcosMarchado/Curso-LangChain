from langchain.vectorstores import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json  # Adicione esta importação no início do arquivo
from datetime import datetime  # Para incluir timestamp no arquivo

# 1. Conexão com o banco
CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5432/juris_db"
embeddings = HuggingFaceEmbeddings(model_name="neuralmind/bert-base-portuguese-cased")

store = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="jurisprudencias_stj"
)

# 2. Pipeline de recomendação com scores reais
def recomendar_jurisprudencia(resumo: str, k=3):
    # 1. Refinar a consulta mantendo contexto jurídico
    prompt = ChatPromptTemplate.from_template("""
    Extraia termos jurídicos precisos deste resumo para busca vetorial:
    Resumo: {resumo}
    Termos jurídicos (max 3, formato: 'termo1 termo2 termo3'):
    """)
    conceitos = (prompt | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"resumo": resumo})
    
    # 2. Busca semântica pura (sem MMR)
    docs_scores = store.similarity_search_with_score(conceitos, k=k*3)  # Busca ampliada
    
    # 3. Filtrar por score mínimo e agrupar por processo
    resultados = []
    limiar_score = 0.0  # Ajuste conforme necessário
    
    for doc, score in docs_scores:
        if score >= limiar_score:
            
            resultados.append({
                "_id": doc.metadata.get("_id", "N/A"),
                "score": float(score),
                "relator": doc.metadata.get("relator", "N/A"),
                "data_julgamento": doc.metadata.get("data_julgamento", "N/A"),
                "numero_registro": doc.metadata.get("numero_registro", "N/A"),
                "numero_processo": doc.metadata.get("numero_processo", "N/A"),
                "descricao_pedido": doc.metadata.get("descricao_pedido", "N/A"),
                "ementa": doc.metadata.get("ementa", "Ementa não disponível"),
                "chunk_id": doc.metadata.get("chunk_num", 0),
                "total_chunks": len(store.similarity_search("", filter={"identificacao": doc.metadata["identificacao"]})),
                "conceitos_busca": conceitos
            })

    
    # Eliminar duplicados e manter os top-k
    resultados = sorted(
        list({v['_id']:v for v in resultados}.values()),
        key=lambda x: x["score"],
        reverse=True
    )[:k]
    
    return resultados

# 3. Exemplo de uso
resumo = """
O processo discute a negativa de seguimento de um recurso especial com base em tese repetitiva do STJ. A parte alegou negativa de prestação jurisdicional, mas o STJ entendeu que, havendo aplicação de precedente vinculante, não há vício. Também foi afastada a possibilidade de análise por divergência jurisprudencial. O agravo interno foi desprovido.
"""
recomendacoes = recomendar_jurisprudencia(resumo)

def salvar_para_json(resultados, nome_arquivo=None):
    if not nome_arquivo:
        data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"recomendacoes_jurisprudencia_{data_hora}.json"
    
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Resultados salvos em {nome_arquivo}")
    return nome_arquivo

print("🎯 Jurisprudências Recomendadas:")
for i, rec in enumerate(recomendacoes, 1):
    print(f"\n{i}. {rec['descricao_pedido']} (Score: {rec['score']:.4f})")
    print(f"   Processo: {rec['numero_processo']}")
    print(f"   Descrição: {rec['descricao_pedido']}")
    print(f"   Relator: {rec['relator']}")
    print(f"   Ementa: {rec['ementa']}")
    print(f"   Conceitos usados: {rec['conceitos_busca']}")


# Salvar em JSON
arquivo_gerado = salvar_para_json(recomendacoes)