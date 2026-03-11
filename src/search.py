import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "semantic_search")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
PGVECTOR_COLLECTION = os.getenv("PGVECTOR_COLLECTION", "document_chunks")

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")

import openai
import logging

LOGLEVEL = os.getenv("LOGLEVEL", "INFO")
level = getattr(logging, LOGLEVEL.upper(), logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(level=level)
logger = logging.getLogger(__name__)
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from langchain_postgres import PGVector
except Exception:
    PGVector = None

try:
    import psycopg
except Exception:
    psycopg = None

if PROVIDER == "gemini" and genai is not None:
    genai.configure(api_key=GOOGLE_API_KEY)


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


class EmbeddingsClient:
    def embed_documents(self, texts):
        if PROVIDER == "openai":
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        if PROVIDER == "gemini":
            if genai is None:
                raise RuntimeError("Biblioteca google.generativeai não disponível para Gemini")
            result = genai.embed_content(model=GOOGLE_EMBEDDING_MODEL, content=texts)
            return result["embedding"] if isinstance(texts, str) else [result["embedding"]]
        raise RuntimeError("Provider não suportado para embeddings")

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def get_embeddings():
    return EmbeddingsClient()


def get_llm():
    def call_llm(prompt_text: str) -> str:
        if PROVIDER == "openai":
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=OPENAI_LLM_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return resp.choices[0].message.content.strip()
        if PROVIDER == "gemini":
            if genai is None:
                raise RuntimeError("Biblioteca google.generativeai não disponível para Gemini")
            gemini_model = genai.GenerativeModel(GOOGLE_LLM_MODEL)
            resp = gemini_model.generate_content(prompt_text)
            return resp.text
        raise RuntimeError("Provider não suportado para LLM")

    return call_llm


class PostgresVectorStoreFallback:
    def __init__(self, embeddings_client: EmbeddingsClient):
        if psycopg is None:
            raise RuntimeError("psycopg não disponível para fallback de armazenamento")
        self.embeddings = embeddings_client
        conn_str = os.getenv("DATABASE_URL") or f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        self.conn = psycopg.connect(conn_str)
        self._ensure_table()

    def _ensure_table(self):
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {PGVECTOR_COLLECTION} (
          id serial PRIMARY KEY,
          content text,
          embedding vector
        );
        """
        with self.conn.cursor() as cursor:
            cursor.execute(create_sql)
            self.conn.commit()

    def add_documents(self, documents):
        with self.conn.cursor() as cursor:
            for document in documents:
                content_text = document.page_content.replace("'", "''")
                embedding = self.embeddings.embed_documents([content_text])[0]
                embedding_literal = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                sql = f"INSERT INTO {PGVECTOR_COLLECTION} (content, embedding) VALUES ('{content_text}', '{embedding_literal}'::vector)"
                cursor.execute(sql)
            self.conn.commit()

    def similarity_search_with_score(self, query, k=10):
        query_embedding = self.embeddings.embed_query(query)
        embedding_literal = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
        sql = f"SELECT content, embedding <-> '{embedding_literal}'::vector AS distance FROM {PGVECTOR_COLLECTION} ORDER BY distance LIMIT {k}"
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        results = []
        for content_text, distance_value in rows:
            results.append(({"page_content": content_text}, float(distance_value)))
        return results


def get_vector_store(embeddings=None):
    if embeddings is None:
        embeddings = get_embeddings()
    connection_string = os.getenv("DATABASE_URL") or f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    if PGVector is not None:
        try:
            return PGVector(embedding_function=embeddings, collection_name=PGVECTOR_COLLECTION, connection_string=connection_string)
        except Exception as e:
            logger.exception("PGVector constructor with embedding_function failed: %s", e)

        try:
            return PGVector(embeddings, collection_name=PGVECTOR_COLLECTION, connection_string=connection_string)
        except Exception as e:
            logger.exception("PGVector constructor with embeddings arg failed: %s", e)

    logger.info("Falling back to SQL-based PostgresVectorStoreFallback for collection '%s'", PGVECTOR_COLLECTION)
    return PostgresVectorStoreFallback(embeddings)


def retrieve_context(query: str) -> str:
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=10)
    if not results:
        return ""
    texts = [r[0]["page_content"] for r in results]
    contexto = "\n\n---\n\n".join(texts)
    return contexto


def answer_question(query: str) -> str:
    contexto = retrieve_context(query)
    if not contexto.strip():
        return "Não tenho informações necessárias para responder sua pergunta."

    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=query)
    llm = get_llm()
    try:
        resp = llm(prompt)
        return resp
    except Exception:
        return "Não tenho informações necessárias para responder sua pergunta."
