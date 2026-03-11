import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

from langchain_postgres import PGVector
from search import get_embeddings, get_vector_store

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "semantic_search")
PGVECTOR_COLLECTION = os.getenv("PGVECTOR_COLLECTION", "document_chunks")

CONNECTION_STRING = (
    os.getenv("DATABASE_URL")
    or f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


def main():
    if not os.path.exists(PDF_PATH):
        print(f"PDF não encontrado: {PDF_PATH}")
        return

    print("Carregando PDF...")
    pdf_loader = PyPDFLoader(PDF_PATH)
    documents = pdf_loader.load()

    print("Dividindo em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    document_chunks = text_splitter.split_documents(documents)

    print(f"Gerando embeddings para {len(document_chunks)} chunks...")
    embeddings_client = get_embeddings()

    print("Persistindo chunks no banco de dados...")

    vector_store = get_vector_store(embeddings_client)
    vector_store.add_documents(document_chunks)

    print("Ingestão finalizada.")


if __name__ == "__main__":
    main()