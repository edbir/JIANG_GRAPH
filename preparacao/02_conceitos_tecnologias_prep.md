# Preparação — Seção 2: Conceitos e Tecnologias

## Objetivo da Seção
Apresentar ao leitor todos os conceitos teóricos e tecnologias necessários para compreender o restante do trabalho. Cada subseção deve ser autocontida o suficiente para que alguém sem conhecimento prévio do tópico consiga acompanhar.

---

## Estrutura Planejada

### 2.1 Retrieval-Augmented Generation (RAG)
**O que cobrir:**
- O problema fundamental: LLMs têm conhecimento estático (cutoff de treinamento) e alucinam
- A solução RAG: recuperar informação relevante de uma base de dados e injetar no contexto do LLM
- O pipeline básico: query → embedding → busca vetorial → chunks relevantes → LLM gera resposta
- Chunking: como texto é dividido em pedaços para indexação
- Limitações: perda de estrutura, sem relações entre entidades, sem causalidade

**Sugestões de estudo:**
- Ler o paper original de RAG (Lewis et al., 2020)
- Pesquisar "how RAG works" — há excelentes explicações visuais em blogs como o da LangChain, Pinecone, e Weaviate
- Entender o conceito de "embedding" e "vector similarity search" (cosine similarity)
- Pesquisar "chunking strategies for RAG" — entender fixed-size, semantic, e sentence-based chunking
- Conceito importante: "context window" dos LLMs e por que não podemos simplesmente jogar todo o texto

### 2.2 Knowledge Graphs
**O que cobrir:**
- Definição: um grafo onde nós representam entidades e arestas representam relações
- A estrutura básica: (sujeito)-[predicado]->(objeto), ou seja, triplas
- Dois paradigmas: RDF (Resource Description Framework) vs. Property Graphs
  - RDF: triplas, SPARQL, ontologias formais (OWL), Linked Data
  - Property Graphs: nós e arestas com propriedades arbitrárias, Cypher como linguagem de consulta
- Por que Property Graphs para este trabalho (flexibilidade de schema, atributos em arestas, Cypher mais natural para traversal)
- O conceito de ontologia: um schema que define os tipos de entidades e relações permitidos
- Reificação: transformar uma relação em um nó para poder anexar metadados (ex: uma Claim como nó intermediário)

**Sugestões de estudo:**
- Pesquisar "knowledge graph introduction" — o artigo da Wikipedia é surpreendentemente bom
- Ler sobre a diferença entre RDF e Property Graphs:
  - Pesquisar "RDF vs Property Graph comparison"
  - O blog da Neo4j tem boas comparações
- Entender o que é uma ontologia no contexto de KGs — pesquisar "ontology in knowledge graphs"
- Pesquisar "graph reification" ou "intermediate node pattern Neo4j" para entender como claims viram nós
- Conceito de ISO GQL (ISO/IEC 39075:2024) — a primeira nova linguagem de banco de dados ISO desde SQL em 1987

### 2.3 GraphRAG
**O que cobrir:**
- A evolução natural: combinar Knowledge Graphs com RAG
- O paper da Microsoft GraphRAG (Edge et al., 2024) como marco:
  - Extração de entidades e relações via LLM
  - Community detection (Leiden) para agrupar entidades em temas
  - Summarization hierárquica das comunidades
  - Dois modos de query: local (entity-centric) e global (community summaries)
- Limitações do GraphRAG da Microsoft para dados conversacionais
- O conceito de "hybrid retrieval": combinar busca vetorial com traversal de grafo
- Outros sistemas relevantes: LightRAG, LazyGraphRAG

**Sugestões de estudo:**
- Ler o paper da Microsoft GraphRAG (arXiv:2404.16130) — focar nas seções 1-3
- Pesquisar "GraphRAG explained" para explicações mais acessíveis
- Ler sobre o conceito de community detection em grafos (o que são comunidades, por que importam)
- Pesquisar "hybrid retrieval RAG" para entender a combinação vetor + grafo
- O paper GraphRAG-Bench (Xiang et al., ICLR 2026) é interessante por mostrar quando GraphRAG NÃO ajuda

### 2.4 Neo4j e Cypher
**O que cobrir:**
- Neo4j como banco de dados de grafos mais popular (property graph)
- Cypher: a linguagem de consulta — pattern matching visual
- Funcionalidades relevantes para o projeto:
  - Vector indexes nativos (Neo4j 5.x+)
  - Full-text indexes
  - MERGE patterns para upserts idempotentes
  - Neo4j GDS (Graph Data Science) para algoritmos como Leiden
- O pacote neo4j-graphrag-python e seus retrievers

**Sugestões de estudo:**
- Fazer o tutorial gratuito "Neo4j Fundamentals" no GraphAcademy (neo4j.com/graphacademy) — é curto e prático
- Aprender Cypher básico: MATCH, CREATE, MERGE, WHERE, RETURN
- Pesquisar "Neo4j vector index" para entender como Neo4j combina grafo + vetores
- Pesquisar "Neo4j GDS Leiden" para entender community detection no Neo4j
- Instalar Neo4j Desktop ou usar Neo4j Aura Free para brincar com queries

### 2.5 Extração de Conhecimento com LLMs
**O que cobrir:**
- O conceito: usar LLMs para extrair entidades e relações de texto não-estruturado
- Single-pass vs. Multi-pass extraction — por que múltiplos passes são superiores
- Structured Outputs: como garantir que o LLM retorne JSON válido (constrained decoding)
- Pydantic como ferramenta para definir schemas de extração
- O conceito de "tiered model selection": modelos baratos para tarefas simples, modelos caros para tarefas complexas
- Entity resolution: o problema de deduplicação (mesma entidade com nomes diferentes)

**Sugestões de estudo:**
- Pesquisar "LLM knowledge graph extraction" — há tutoriais práticos
- Ler sobre OpenAI Structured Outputs — a documentação oficial é clara
- Pesquisar "Pydantic tutorial" se não conhece — é a base dos schemas de extração
- Ler sobre entity resolution / entity matching — pesquisar "entity resolution knowledge graphs"
- O paper KGGen (NeurIPS 2025) é uma boa referência para multi-stage extraction
- O paper SAC-KG (ACL 2024) para o pipeline Generator→Verifier→Pruner
- Pesquisar "coreference resolution NLP" para entender o problema de pronomes e referências

### 2.6 Community Detection e o Algoritmo de Leiden
**O que cobrir:**
- O que é community detection: encontrar grupos de nós densamente conectados
- O algoritmo de Leiden como evolução do Louvain — por que é preferido
- Comunidades hierárquicas: diferentes níveis de granularidade
- A aplicação em GraphRAG: comunidades como "temas" que podem ser sumarizados por LLM
- O papel do parâmetro gamma (resolução) e do parâmetro de seed para incrementalidade

**Sugestões de estudo:**
- Pesquisar "community detection in graphs explained" — conceito geral
- Ler sobre o algoritmo de Louvain primeiro (é mais simples) e depois Leiden como melhoria
- O paper original do Leiden: Traag et al., 2019 — "From Louvain to Leiden"
- Pesquisar "Leiden algorithm Neo4j GDS" para ver como é usado na prática
- Entender o conceito de "modularity" em grafos — é a métrica que Leiden otimiza

### 2.7 Model Context Protocol (MCP)
**O que cobrir:**
- O que é MCP: um protocolo aberto para conectar LLMs a fontes de dados externas
- Anunciado pela Anthropic em novembro de 2024, agora sob a Linux Foundation
- Os três primitivos: Tools, Resources, Prompts — sobre JSON-RPC 2.0
- Por que MCP para este projeto: permite que qualquer LLM client-side (Claude, GPT) consulte o knowledge graph
- A diferença entre expor o grafo via API REST vs. via MCP (o LLM pode escolher qual tool usar)

**Sugestões de estudo:**
- Ler a documentação oficial do MCP (modelcontextprotocol.io)
- Pesquisar "MCP protocol explained" para visões gerais
- Entender o conceito de JSON-RPC 2.0 em alto nível
- Pesquisar "Neo4j MCP server" — a Neo4j tem servidores MCP oficiais
- Pesquisar "FastMCP Python" — é a biblioteca que usaremos para criar o servidor

---

## Referências-Chave dos Documentos Base
Estas referências já aparecem na pesquisa e serão úteis para esta seção:

**RAG:**
- Lewis et al., 2020 — Paper original de RAG

**Knowledge Graphs:**
- ISO/IEC 39075:2024 — GQL standard
- Neo4j Intermediate Node Pattern (reificação)
- Angles et al., ACM SIGMOD 2024 — RDF→property graph transformations

**GraphRAG:**
- Edge et al., 2024 — Microsoft GraphRAG (arXiv:2404.16130)
- Guo et al., 2024 — LightRAG (HKU)
- Xiang et al., ICLR 2026 — GraphRAG-Bench

**Extração:**
- Mo et al., NeurIPS 2025 — KGGen
- Chen et al., ACL 2024 — SAC-KG
- ODKE+ — 98.8% precision
- OpenAI Structured Outputs documentation

**Community Detection:**
- Traag et al., 2019 — Leiden algorithm
- Sahu, 2024 — Dynamic Frontier Leiden (arXiv:2405.11658)

**MCP:**
- Anthropic MCP specification (v2025-11-25)
- neo4j-contrib/mcp-neo4j (GitHub)

**Conversational Data:**
- Maharana et al., ACL 2024 — LoCoMo
- Rook, ISWC 2025 — OKE-D
- Rasmussen et al., 2025 — Graphiti/Zep (arXiv:2501.13956)
