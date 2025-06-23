from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector

CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5432/juris_db"
embeddings = HuggingFaceEmbeddings(model_name="neuralmind/bert-base-portuguese-cased")

# Teste de embedding
test_text = "Habeas corpus não conhecido. Ordem concedida de ofício para revogar a prisão preventiva"
test_embed = embeddings.embed_query(test_text)
print(f"Embedding length: {len(test_embed)}")  # Deve ser 768

store = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="jurisprudencias_stj"
)

# Busca direta sem MMR para teste
docs_scores = store.similarity_search_with_score(test_text, k=3)
for doc, score in docs_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}...")