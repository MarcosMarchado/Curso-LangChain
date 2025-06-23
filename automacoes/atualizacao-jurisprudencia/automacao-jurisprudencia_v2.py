from sqlalchemy import create_engine, text
from langchain.vectorstores import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from datetime import datetime
import json

# Configuração
CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5432/juris_db"
engine = create_engine(CONNECTION_STRING)

# Embeddings e PGVector
embeddings = HuggingFaceEmbeddings(model_name="neuralmind/bert-base-portuguese-cased")
store = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="jurisprudencias_stj"
)

def buscar_hibrido(consulta: str, k=10, alpha=0.5):
    print("🔎 Executando busca híbrida...")

    # Busca semântica
    semanticos = store.similarity_search_with_score(consulta, k=k*2)
    resultados_sem = {_doc.metadata["_id"]: {"doc": _doc, "score_sem": s} for _doc, s in semanticos}

    # Busca lexical via FTS | FTS significa Full-Text Search — ou Busca de Texto Completa.
    with engine.connect() as conn:
        sql = text("""
            SELECT cmetadata, document
            FROM langchain_pg_embedding
            WHERE to_tsvector('portuguese', cmetadata->>'ementa') @@ plainto_tsquery('portuguese', :query)
            LIMIT :limite
        """)
        rows = conn.execute(sql, {"query": consulta, "limite": k*5}).fetchall()

    resultados_lex = {}
    for row in rows:
        _id = row[0].get("_id")
        if _id:
            resultados_lex[_id] = {"metadata": row[0], "page_content": row[1], "score_lex": 1.0}

    # Combinar e rerankear
    combinados = {}
    ids = set(resultados_sem) | set(resultados_lex)

    for _id in ids:
        score_sem = resultados_sem.get(_id, {}).get("score_sem", 0.0)
        score_lex = resultados_lex.get(_id, {}).get("score_lex", 0.0)
        score_total = alpha * score_sem + (1 - alpha) * score_lex

        doc = resultados_sem.get(_id, {}).get("doc")
        if not doc and _id in resultados_lex:
            from langchain.docstore.document import Document
            doc = Document(
                page_content=resultados_lex[_id]["page_content"],
                metadata=resultados_lex[_id]["metadata"]
            )

        if doc:
            combinados[_id] = {
                "doc": doc,
                "score_total": score_total,
                "score_sem": score_sem,
                "score_lex": score_lex
            }

    resultados_finais = sorted(combinados.values(), key=lambda x: x["score_total"], reverse=True)[:k]
    return resultados_finais



def recomendar_jurisprudencia(resumo: str, k=10, alpha=0.5):
    prompt = ChatPromptTemplate.from_template("""
    Extraia termos jurídicos precisos deste resumo para busca vetorial:
    Resumo: {resumo}
    Termos jurídicos (max 10, formato: 'termo1 termo2 termo3'):
    """)
    conceitos = (prompt | ChatOpenAI(temperature=0) | StrOutputParser()).invoke({"resumo": resumo})
    
    # ✅ AGORA usando busca híbrida!
    docs_rerank = buscar_hibrido(conceitos, k=k, alpha=alpha)

    resultados = []
    for r in docs_rerank:
        doc = r["doc"]
        resultados.append({
            "_id": doc.metadata.get("_id", "N/A"),
            "score": float(r["score_total"]),
            "relator": doc.metadata.get("relator", "N/A"),
            "data_julgamento": doc.metadata.get("data_julgamento", "N/A"),
            "numero_registro": doc.metadata.get("numero_registro", "N/A"),
            "numero_processo": doc.metadata.get("numero_processo", "N/A"),
            "descricao_pedido": doc.metadata.get("descricao_pedido", "N/A"),
            "ementa": doc.metadata.get("ementa", "Ementa não disponível"),
            "conceitos_busca": conceitos
        })

    return resultados

def salvar_para_json(resultados, nome_arquivo=None):
    if not nome_arquivo:
        data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"recomendacoes_jurisprudencia_{data_hora}.json"
    
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Resultados salvos em {nome_arquivo}")
    return nome_arquivo

# ✅ Teste
# resumo = """
# Prisão preventiva por tráfico de drogas e posse ilegal de arma de fogo.
# """

# resumo = """
# Habeas corpus contra aumento da pena em homicídio qualificado e tentado.
# Discussão sobre fundamentação da dosimetria e ausência de continuidade delitiva.
# Decisão mantém agravamento da pena e rejeita reexame de provas.
# """

resumo = """
Habeas corpus contra negativa do Ministério Público para acordo de não persecução penal (ANPP).
Discussão sobre requisitos legais e soma das penas em concurso material acima do limite legal.
Decisão confirma discricionariedade do MP e nega a ordem por ausência de constrangimento ilegal.
"""
recomendacoes = recomendar_jurisprudencia(resumo, k=10, alpha=0.5)

for i, rec in enumerate(recomendacoes, 1):
    print(f"\n{i}. (Score: {rec['score']:.4f})")
    print(f"   Processo: {rec['numero_processo']}")
    print(f"   Descrição: {rec['descricao_pedido']}")
    print(f"   Relator: {rec['relator']}")
    print(f"   Ementa: {rec['ementa'][:500]}...")
    print(f"   Conceitos extraídos: {rec['conceitos_busca']}")

salvar_para_json(recomendacoes)
