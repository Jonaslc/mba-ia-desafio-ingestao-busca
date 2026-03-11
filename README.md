# Desafio MBA - Ingestão e Busca Semântica (PDF -> PostgreSQL + pgvector)

Visão rápida: ingest PDF, gerar embeddings e buscar por similaridade em Postgres+pgvector.

Tecnologias: Python, LangChain (utilitários), OpenAI (opcional), PostgreSQL + pgvector, Docker Compose.

Estrutura:
- `docker-compose.yml` - Postgres com pgvector
- `.env.example` - variáveis de ambiente
- `src/ingest.py` - ingere `document.pdf` e persiste vetores
- `src/search.py` - recuperação e chamada ao LLM
- `src/chat.py` - CLI de perguntas

Instalação e uso:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
python src/ingest.py
python src/chat.py
```

Observação: as respostas são geradas exclusivamente com base no contexto recuperado do PDF. Se a informação não estiver no contexto, o sistema responde exatamente:

"Não tenho informações necessárias para responder sua pergunta."

Parar containers (no diretório com `docker-compose.yml`):

```bash
# parar containers (mantém volumes/networks)
docker compose stop

# parar e remover containers + networks (mantém volumes)
docker compose down

# parar e remover tudo incluindo volumes (use com cuidado)
docker compose down -v

# ver status
docker compose ps
```

Logs e depuração
-----------------

- Logs do Python aparecem no terminal onde você executa `python src/chat.py` ou `python src/ingest.py`.
- Para ver mensagens do `PGVector` constructor falhando (fallback), observe mensagens `INFO`/`ERROR` no início da execução.
- Para aumentar a verbosidade, exporte `PYTHONLOGGING` ou rode com `python -u -c "import logging; logging.basicConfig(level=logging.DEBUG); import src.chat"`, ou simplesmente defina a variável de ambiente `LOGLEVEL=DEBUG` e adapte o script se desejar.
 - Para aumentar a verbosidade, exporte `PYTHONLOGGING` ou rode com `python -u -c "import logging; logging.basicConfig(level=logging.DEBUG); import src.chat"`, ou simplesmente defina a variável de ambiente `LOGLEVEL=DEBUG` e adapte o script se desejar.

Exemplo rápido:

```bash
# ativar logs DEBUG apenas para a execução atual
LOGLEVEL=DEBUG python src/chat.py
```

Ver logs do Postgres (container):

```bash
docker compose logs -f postgres
```