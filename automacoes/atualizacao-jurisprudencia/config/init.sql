-- Ativa a extensão PGVector (necessária para armazenar vetores e usar índices vetoriais)
CREATE EXTENSION IF NOT EXISTS vector;

-- Cria a tabela para armazenar os embeddings e metadados das jurisprudências
CREATE TABLE IF NOT EXISTS jurisprudencias_stj (
    id SERIAL PRIMARY KEY,         -- Chave primária incremental
    content TEXT,                  -- Conteúdo textual do chunk da ementa
    embedding VECTOR(768),         -- Vetor de embedding com 768 dimensões (ex: BERTimbau)
    metadata JSONB                 -- Metadados do documento (relator, número processo, etc), no formato JSONB
);

-- Cria tabela langchain_pg_collection
CREATE TABLE IF NOT EXISTS public.langchain_pg_collection
(
    uuid uuid NOT NULL,
    name character varying COLLATE pg_catalog."default" NOT NULL,
    cmetadata json,
    CONSTRAINT langchain_pg_collection_pkey PRIMARY KEY (uuid),
    CONSTRAINT langchain_pg_collection_name_key UNIQUE (name)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.langchain_pg_collection
    OWNER to postgres;


-- Cria a tabela langchain_pg_embedding
CREATE TABLE IF NOT EXISTS public.langchain_pg_embedding
(
    id character varying COLLATE pg_catalog."default" NOT NULL,
    collection_id uuid,
    embedding vector,
    document character varying COLLATE pg_catalog."default",
    cmetadata jsonb,
    CONSTRAINT langchain_pg_embedding_pkey PRIMARY KEY (id),
    CONSTRAINT langchain_pg_embedding_collection_id_fkey FOREIGN KEY (collection_id)
        REFERENCES public.langchain_pg_collection (uuid) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.langchain_pg_embedding
    OWNER to postgres;
-- Index: ix_cmetadata_gin

-- DROP INDEX IF EXISTS public.ix_cmetadata_gin;

CREATE INDEX IF NOT EXISTS ix_cmetadata_gin
    ON public.langchain_pg_embedding USING gin
    (cmetadata jsonb_path_ops)
    TABLESPACE pg_default;

-- Cria um índice vetorial usando IVFFlat com similaridade cosseno
-- Esse índice permite buscas aproximadas eficientes com "ORDER BY embedding <-> vetor"
-- 'vector_cosine_ops' define que a métrica de distância será a distância cosseno
-- 'lists = 100' divide os vetores em 100 clusters (inverted lists) para balancear performance e precisão
-- Sugestão: escolha um valor de lists próximo a sqrt(N), onde N é a quantidade total esperada de vetores
-- Exemplo: para ~10.000 vetores, usar lists ≈ 100 é uma boa escolha

--Índice vetorial para busca semântica com distância cosseno
CREATE INDEX IF NOT EXISTS idx_embedding_vector
    ON langchain_pg_embedding
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100); -- ~√n, para n ≈ 10.000 vetores


-- Índice FTS para buscas tradicionais por texto (palavra-chave)
Permite buscar em português usando stemming e stopwords corretos
CREATE INDEX IF NOT EXISTS idx_juris_fts
    ON langchain_pg_embedding
    USING gin(to_tsvector('portuguese', cmetadata->>'ementa'));

--Indice do tipo IVFFlat (Inverted File with Flat Compression), um algoritmo eficiente para buscas aproximadas em vetores.
--HNSW (outro algoritmo de índice): Mais preciso que IVFFlat, mas consome mais memória. Disponível em extensões como lantern.