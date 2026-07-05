# KUB RAG Assistant Model

Logfire observability is wired into the ingestion entrypoint with the target project
`lakshayraj505/starter-project` on `https://logfire-us.pydantic.dev`.

Local setup:

```bash
uv run logfire --base-url='https://logfire-us.pydantic.dev' auth
uv run logfire --base-url='https://logfire-us.pydantic.dev' projects use --org 'lakshayraj505' 'starter-project'
```

Run ingestion with:

```bash
uv run python scripts/run_ingestion.py
```

If you deploy or run in CI, set `LOGFIRE_TOKEN` in the environment.
