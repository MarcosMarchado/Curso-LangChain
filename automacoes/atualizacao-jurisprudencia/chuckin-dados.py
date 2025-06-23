from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import time

# 1. Configuração do Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 2. Carregar apenas as primeiras 10 linhas do dataset
try:
    ds = load_dataset("celsowm/jurisprudencias_stj", split='train[:100]')
    print(f"Total de registros a processar: {len(ds)}")
except Exception as e:
    print(f"Erro ao carregar dataset: {e}")
    exit()

# 3. Configurar Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="neuralmind/bert-base-portuguese-cased",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 4. Conexão com o Postgres
CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5432/juris_db"

# 5. Criar a store PGVector primeiro
store = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="jurisprudencias_stj"
)

# 6. Processar registros
for i, record in enumerate(ds):
    try:
        if not record.get("ementa"):
            print(f"Registro {i} sem ementa. Pulando...")
            continue
            
        chunks = text_splitter.split_text(record["ementa"])
        
        for chunk_num, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "_id": record.get("_id", ""),
                    "identificacao": record.get("identificacao", ""),
                    "relator": record.get("relator", ""),
                    "data_julgamento": record.get("data_julgamento", ""),
                    "numero_registro": record.get("numero_registro", ""),
                    "descricao_pedido": record.get("descricao_pedido", ""),
                    "numero_processo": record.get("numero_processo", ""),
                    "ementa": record.get("ementa", ""),
                    "chunk_num": chunk_num + 1
                }
            )
            
            # Inserir documento individualmente
            store.add_documents([doc])
            print(f"✅ Chunk {chunk_num+1} do registro {i+1} inserido (ID: {record.get('_id', '')})")
            
        time.sleep(0.5)  # Pausa curta entre registros
        
    except Exception as e:
        print(f"❌ Erro no registro {i+1}: {str(e)}")

print("\nProcesso finalizado!")