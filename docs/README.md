# Documentação Técnica do Projeto

Esta pasta contém documentação técnica do sistema. Seu propósito é duplo:

1. **Para o humano (Eduardo):** Entender o que cada parte faz, por que existe, e como se conecta ao resto.
2. **Para uma IA futura (Kiro ou outro):** Recontextualizar-se rapidamente no projeto sem precisar reler todo o código.

## Como usar esta documentação

Se você é uma IA começando a trabalhar neste projeto:
1. Leia `PROJECT_OVERVIEW.md` primeiro — dá a visão geral completa
2. Depois leia o doc específico do módulo que precisa modificar
3. Os docs em `design/` contêm o design thinking detalhado (o "porquê" de cada decisão)
4. Os docs na raiz de `docs/` são referências técnicas rápidas (o "como")

## Índice

### Visão Geral
- **`PROJECT_OVERVIEW.md`** — Arquitetura completa, fluxo de dados, decisões, status

### Referências Técnicas (como cada módulo funciona)
- **`config.md`** — Configuração centralizada (Pydantic Settings, .env)
- **`text_cleaner.md`** — Limpeza de transcrições
- **`chunker.md`** — Two-pass chunking (LLM segmentation + token split)
- **`schemas.md`** — Modelos Pydantic para structured outputs
- **`prompts.md`** — Prompts de extração
- **`extractor.md`** — Orquestrador multi-pass
- **`resolver.md`** — Entity resolution cascade
- **`neo4j_client.md`** — Conexão, constraints, indexes
- **`upsert.md`** — MERGE patterns para inserção
- **`communities.md`** — Leiden community detection + summarization
- **`retrieval.md`** — Query classification, vector search, Text-to-Cypher, RRF
- **`mcp_server.md`** — MCP server com 6 tools
- **`ingest_episode.md`** — Script de ingestão end-to-end

### Design Thinking (por que cada módulo é como é)
- **`design/04_schemas_extraction_design.md`**
- **`design/05_prompts_extraction_design.md`**
- **`design/06_chunker_design.md`**
- **`design/07_extractor_design.md`**
- **`design/08_resolver_design.md`**
- **`design/09_upsert_design.md`**
- **`design/10_communities_design.md`**
- **`design/11_retrieval_design.md`**
