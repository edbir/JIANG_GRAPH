# Preparação — Seção 4: Desenvolvimento

## Objetivo da Seção
Descrever em detalhe O QUE foi construído, COMO funciona, e POR QUE cada decisão foi tomada. Esta é a seção mais longa do TCC — é onde você demonstra competência técnica. A implementação já está pronta; esta seção a documenta para o artigo.

---

## Estrutura Planejada

### 4.1 Visão Geral da Arquitetura
**O que cobrir:**
- O diagrama de 4 camadas (Client → MCP Server → Retrieval → Neo4j ← Ingestion)
- O fluxo de dados end-to-end (transcript → graph → query → response)
- A decisão de construir custom vs. usar framework (com justificativa quantitativa)
- Stack tecnológico: Python, Neo4j, OpenAI API, MCP

**Base na implementação:**
- Ver `docs/PROJECT_OVERVIEW.md` para o diagrama e fluxo
- Ver `preparacao/03_infraestrutura_e_dependencias.md` para decisões de stack

**Sugestões de estudo:**
- Pesquisar como outros TCCs descrevem arquiteturas de software
- Diagramas UML de componentes podem ser úteis aqui

### 4.2 Schema do Knowledge Graph
**O que cobrir:**
- Os 3 princípios de design: conversation-centric hierarchy, claims as first-class nodes, bi-temporal metadata
- Os 9 tipos de nó (Episode, Segment, Utterance, Person, Organization, Topic, Event, Claim, Community)
- Os tipos de relação e seus significados
- Constraints e indexes (por que cada um existe)
- Escala esperada (~134 Episodes, ~30K Utterances, ~3-8K Claims)

**Base na implementação:**
- Ver `src/extraction/schemas.py` para os tipos de entidade
- Ver `src/graph/neo4j_client.py` para constraints e indexes
- Ver `docs/design/04_schemas_extraction_design.md` para decisões de design

**Sugestões de estudo:**
- Pesquisar "property graph data modeling" e "Neo4j graph modeling best practices"
- Entender o conceito de reificação (por que claims são nós e não propriedades)
- Ler sobre o intermediate node pattern do Neo4j

### 4.3 Pipeline de Preprocessing
**O que cobrir:**
- Text cleaning: o que é removido e por quê (fillers, artifacts, stutters)
- Two-pass chunking: Level 1 (LLM segmentation) + Level 2 (token-bounded split)
- Por que LLM para segmentação (superioridade sobre heurísticas)
- O papel do overlap (cross-boundary assertions, referência ao Rook ISWC 2025)
- Parâmetros escolhidos (1500 tokens, 3 sentenças overlap, 20 sentenças/janela)

**Base na implementação:**
- Ver `src/preprocessing/text_cleaner.py`
- Ver `src/preprocessing/chunker.py`
- Ver `docs/design/06_chunker_design.md`

**Sugestões de estudo:**
- Pesquisar "chunking strategies for RAG" — comparar com nossa abordagem
- Entender por que 1500 tokens é o sweet spot (contexto vs. custo)
- O paper Rook (ISWC 2025) sobre span distances em diálogo

### 4.4 Pipeline de Extração Multi-Pass
**O que cobrir:**
- Por que multi-pass é superior a single-pass (ECR +16%, KGGen +18%, SAC-KG 89.32%)
- Os 5 passes em detalhe: o que cada um faz, qual modelo usa, por quê
- Structured Outputs: como garantem 100% conformidade JSON
- Tiered model selection: gpt-4o-mini para volume, gpt-4o para nuance
- Prompt engineering: schema-constrained, domain-specific, context-enriched
- Caching: idempotência e economia

**Base na implementação:**
- Ver `src/extraction/schemas.py` (o que cada pass produz)
- Ver `src/extraction/prompts.py` (como pedimos ao LLM)
- Ver `src/extraction/extractor.py` (orquestração)
- Ver `docs/design/04_schemas_extraction_design.md`, `05_prompts_extraction_design.md`, `07_extractor_design.md`

**Sugestões de estudo:**
- Ler sobre OpenAI Structured Outputs (documentação oficial)
- Entender constrained decoding (por que garante conformidade)
- O paper KGGen (NeurIPS 2025) para multi-stage extraction
- Pesquisar "prompt engineering for information extraction"

### 4.5 Entity Resolution
**O que cobrir:**
- O problema: mesma entidade com nomes diferentes
- A cascata de 4 estágios (exact → fuzzy → embedding → LLM)
- Percentuais de resolução por estágio (~40%, ~35%, ~20%, ~5%)
- O princípio conservador (falso positivo > falso negativo em KGs)
- Transitive closure no merge map

**Base na implementação:**
- Ver `src/extraction/resolver.py`
- Ver `docs/design/08_resolver_design.md`

**Sugestões de estudo:**
- Pesquisar "entity resolution" e "entity matching"
- O paper CORE-KG (28.32% mais duplicação sem coreference)
- Pesquisar "RapidFuzz" para entender fuzzy string matching

### 4.6 Inserção no Grafo (Upsert)
**O que cobrir:**
- O padrão MERGE do Neo4j (find or create)
- A regra de ouro: bind nodes first, then MERGE relationships
- Ordem de inserção (Episode → Segment → Utterance → Entity → Claim → Relationships)
- Enriquecimento progressivo (descrições melhoram com mais episódios)
- IDs determinísticos para idempotência

**Base na implementação:**
- Ver `src/graph/upsert.py`
- Ver `docs/design/09_upsert_design.md`

**Sugestões de estudo:**
- Pesquisar "Neo4j MERGE pattern" e "idempotent graph operations"
- Entender ON CREATE SET vs ON MATCH SET

### 4.7 Community Detection e Summarization
**O que cobrir:**
- O algoritmo de Leiden (otimiza modularidade, garante conectividade)
- Hierarquia de comunidades (múltiplos níveis de granularidade)
- Summarization via LLM (membros + relações → parágrafo descritivo)
- Incrementalidade (seeded Leiden para updates eficientes)
- O papel no retrieval (queries temáticas)

**Base na implementação:**
- Ver `src/graph/communities.py`
- Ver `docs/design/10_communities_design.md`

**Sugestões de estudo:**
- O paper original do Leiden (Traag et al., 2019)
- Como Microsoft GraphRAG usa Leiden para community summaries
- Pesquisar "Neo4j GDS Leiden" para a implementação prática

### 4.8 Camada de Retrieval
**O que cobrir:**
- Query classification (4 tipos, roteamento adaptativo)
- Vector search + graph fan-out (o padrão dual-path)
- Text-to-Cypher com self-healing (flexibilidade + robustez)
- Reciprocal Rank Fusion (fusão de múltiplas estratégias)
- Context assembly (formatação estruturada para o LLM)

**Base na implementação:**
- Ver `src/retrieval/classifier.py`, `vector_search.py`, `text2cypher.py`, `assembler.py`
- Ver `docs/design/11_retrieval_design.md`

**Sugestões de estudo:**
- Adaptive-RAG (Jeong et al., NAACL 2024) — query-adaptive retrieval
- RRF (Cormack et al., SIGIR 2009) — a fórmula de fusão
- Neo4j Text2Cypher documentation

### 4.9 MCP Server
**O que cobrir:**
- O que é MCP e por que foi escolhido como interface
- Os 6 tools e seus propósitos
- A separação: server retorna dados, client LLM gera texto
- Como configurar um client para conectar ao server

**Base na implementação:**
- Ver `src/mcp/server.py`
- Ver `docs/mcp_server.md`

**Sugestões de estudo:**
- Documentação oficial do MCP (modelcontextprotocol.io)
- Pesquisar "MCP server Python" para exemplos

---

## Dica para a escrita
A seção 4 deve ser escrita de forma que alguém possa REPRODUZIR o sistema lendo apenas o artigo. Cada decisão deve ter justificativa (preferencialmente com referência a um paper ou benchmark). Use diagramas, trechos de código, e exemplos concretos.
