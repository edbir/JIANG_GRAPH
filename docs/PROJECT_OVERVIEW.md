# Project Overview — GraphRAG para Aulas de História

## O que é este projeto

Um sistema GraphRAG custom que transforma ~134 episódios de aulas de história do YouTube (canal "Predictive History", Prof. Jiang) em um knowledge graph queryable no Neo4j, exposto via MCP server para LLMs.

## Para quem é

Este é o TCC (Trabalho de Conclusão de Curso) de Eduardo Birkheuer em Ciência da Computação na UFRGS.

## O domínio

Prof. Jiang dá aulas de ~1 hora sobre história e geopolítica em inglês. Ele é altamente opinativo: mistura fatos históricos com interpretações pessoais, análises causais, e previsões geopolíticas modernas. Algumas previsões já tiveram desfecho ao longo dos anos do canal.

## Por que GraphRAG (e não RAG tradicional)

RAG tradicional (vector-based) trata texto como chunks planos, perdendo:
- Relações causais entre eventos
- Conexões entre entidades históricas
- A distinção entre fato, opinião, e previsão
- A evolução temporal de previsões

Um knowledge graph preserva tudo isso como estrutura explícita e queryable.

## Por que custom (e não um framework)

Nenhum framework existente (Microsoft GraphRAG, LightRAG, Graphiti) suporta nativamente:
- Modelagem de claims como nós first-class (reificação)
- Tracking temporal de previsões geopolíticas
- Ontologia específica para análise histórica
- O volume e custo adequados para um projeto acadêmico

## Arquitetura em 4 camadas

```
┌─────────────────────────────────────────────────────┐
│ CLIENT LAYER                                         │
│ Claude / GPT via MCP ←→ System prompts per query    │
└────────────────────────────┬────────────────────────┘
                             │ MCP Protocol (JSON-RPC 2.0)
┌────────────────────────────▼────────────────────────┐
│ MCP SERVER                                           │
│ Tools: search, find_claims, compare, timeline,      │
│        run_cypher, get_schema                        │
└────────────────────────────┬────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────┐
│ RETRIEVAL LAYER                                      │
│ Query Classifier → Strategy Router                   │
│ ├── Vector + Graph Fan-out (factual queries)        │
│ ├── Cypher Templates (comparative, temporal)        │
│ ├── Text-to-Cypher (analytical, ad-hoc)            │
│ └── Community Summaries (thematic queries)          │
│ Context Assembler → Reciprocal Rank Fusion          │
└────────────────────────────┬────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────┐
│ NEO4J GRAPH                                          │
│ Vector indexes (chunk + entity embeddings)           │
│ Full-text indexes                                    │
│ Graph: Episode→Segment→Utterance→Entity→Claim       │
│ Community nodes (Leiden hierarchical)                │
└────────────────────────────┬────────────────────────┘
                             ▲
┌────────────────────────────┴────────────────────────┐
│ INGESTION PIPELINE                                   │
│ Transcript → Clean → Chunk (LLM segmentation)       │
│ → Pass 1: Entity extraction (GPT-4o-mini)           │
│ → Pass 2: Relationship extraction (GPT-4o-mini)     │
│ → Pass 3: Claim/opinion extraction (GPT-4o)         │
│ → Pass 4: Coreference resolution (GPT-4o)           │
│ → Entity resolution cascade                         │
│ → MERGE into Neo4j → Embed → Communities            │
└─────────────────────────────────────────────────────┘
```

## Estrutura de pastas

```
src/
├── config.py                    # Configuração centralizada (lê .env)
├── preprocessing/
│   ├── text_cleaner.py          # Limpeza de transcrição (fillers, artifacts)
│   └── chunker.py               # Two-pass chunking (LLM segmentation + token split)
├── extraction/
│   ├── schemas.py               # Pydantic models (o que cada pass produz)
│   ├── prompts.py               # Prompts de extração (o que perguntar ao LLM)
│   ├── extractor.py             # Orquestrador multi-pass (chama a API)
│   └── resolver.py              # Entity resolution cascade (deduplicação)
├── graph/
│   ├── neo4j_client.py          # Conexão, constraints, indexes
│   ├── upsert.py                # MERGE patterns (inserção idempotente)
│   └── communities.py           # Leiden community detection + summarization
├── retrieval/
│   ├── classifier.py            # Classifica tipo de query
│   ├── vector_search.py         # Vector + graph fan-out
│   ├── text2cypher.py           # Natural language → Cypher
│   └── assembler.py             # Context assembly + RRF
└── mcp/
    └── server.py                # MCP server (6 tools)
```

## Decisões técnicas chave

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| LLM provider | OpenAI exclusivo | Simplicidade, elimina variabilidade |
| Modelos | gpt-4o-mini (volume) + gpt-4o (nuance) | Tiered: barato para simples, forte para complexo |
| Embeddings | text-embedding-3-small (1536d) | Custo-efetivo, boa qualidade |
| Banco de grafos | Neo4j 5.x + GDS | Vector indexes nativos, Leiden, Cypher |
| Chunking Level 1 | LLM-based | Superior a heurísticas para detecção de tópico |
| Extração | Multi-pass (5 passes) | +16-18% sobre single-pass (ECR, KGGen) |
| Entity resolution | Cascata 4 estágios | 95% resolvido sem LLM, conservador |
| Retrieval | Hybrid (vector + graph + T2C) | Cada tipo de query tem estratégia ótima |
| Interface | MCP server | Qualquer LLM pode consultar o grafo |
| Transcrição | Groq free tier (MVP) | Não é foco da pesquisa |

## Fluxo de dados (end-to-end)

```
YouTube audio
    → Groq/Whisper (transcrição)
    → text_cleaner (limpeza)
    → chunker Level 1 (LLM segmenta por tópico)
    → chunker Level 2 (split em ~1500 tokens com overlap)
    → extractor Pass 1 (entidades)
    → extractor Pass 2 (relações)
    → extractor Pass 3 (claims)
    → extractor Pass 4 (coreference, janela de chunks)
    → resolver (deduplicação de entidades)
    → upsert (MERGE no Neo4j)
    → embeddings (gerar e armazenar)
    → communities (Leiden + summarization)
    → PRONTO PARA QUERIES
```

## Schema do grafo Neo4j

### Node types:
- **Episode** — um episódio do canal
- **Segment** — bloco temático dentro de um episódio (~10-15 min)
- **Utterance** — chunk de texto (~1500 tokens) — unidade de busca vetorial
- **Person** — figuras históricas, líderes, filósofos
- **Organization** — impérios, nações, dinastias, religiões
- **Topic** — conceitos, ideologias, regiões geográficas
- **Event** — guerras, batalhas, tratados, revoluções
- **Claim** — interpretação, previsão, ou análise causal do Jiang
- **Community** — cluster temático detectado por Leiden

### Key relationships:
- Episode → HAS_SEGMENT → Segment → HAS_CHUNK → Utterance
- Utterance → MENTIONS → Entity (Person/Org/Topic/Event)
- Claim → EXTRACTED_FROM → Utterance
- Claim → ABOUT → Entity
- Claim → SUPPORTS/CONTRADICTS → Claim
- Claim → REVISES_PREDICTION → Claim
- Person → ROLE_IN → Organization
- Event → LED_TO/CAUSED → Event
- Entity → IN_COMMUNITY → Community

## Diretrizes de desenvolvimento

Ver `DEVELOPMENT_GUIDELINES.md` na raiz. Resumo:
- Desenvolvimento incremental (um passo de cada vez)
- O aluno lidera decisões
- Documentar cada etapa no TCC.md
- Sem pressa — compreensão > velocidade

## Status atual

O pipeline de ingestion está implementado em código (preprocessing + extraction + resolver).
Falta: graph upsert, communities, retrieval layer, MCP server.
O artigo (TCC.md) tem estrutura definida mas não está escrito.
