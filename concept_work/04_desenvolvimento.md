# 4. DESENVOLVIMENTO

Esta seção descreve a arquitetura, implementação e decisões técnicas do sistema proposto.

## 4.1 Visão Geral da Arquitetura

O sistema é organizado em quatro camadas com responsabilidades bem definidas:

```
┌─────────────────────────────────────────────────────────┐
│ CAMADA CLIENT                                            │
│ Claude / GPT via MCP ←→ System prompts per query type   │
└────────────────────────────────┬────────────────────────┘
                                 │ MCP Protocol (JSON-RPC 2.0)
┌────────────────────────────────▼────────────────────────┐
│ MCP SERVER                                               │
│ Tools: search_lectures, find_claims, compare_entities,  │
│        topic_timeline, run_cypher, get_schema            │
└────────────────────────────────┬────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────┐
│ CAMADA DE RETRIEVAL                                      │
│ Query Classifier → Strategy Router                       │
│ ├── Vector + Graph Fan-out (queries factuais)           │
│ ├── Cypher Templates (comparativas, temporais)          │
│ ├── Text-to-Cypher (analíticas, ad-hoc)                │
│ └── Community Summaries (queries temáticas)             │
│ Context Assembler → Reciprocal Rank Fusion              │
└────────────────────────────────┬────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────┐
│ NEO4J GRAPH                                              │
│ Vector indexes (chunk + entity embeddings)               │
│ Full-text indexes                                        │
│ Grafo: Episode→Segment→Utterance→Entity→Claim           │
│ Community nodes (Leiden hierárquico)                      │
└────────────────────────────────┬────────────────────────┘
                                 ▲
┌────────────────────────────────┴────────────────────────┐
│ PIPELINE DE INGESTÃO                                     │
│ YouTube → yt-dlp → Groq Whisper → Transcript            │
│ → Clean → Chunk (LLM segmentation + token split)         │
│ → Pass 1: Entity extraction (GPT-4o-mini)               │
│ → Pass 2: Relationship extraction (GPT-4o-mini)         │
│ → Pass 3: Claim/opinion extraction (GPT-4o)             │
│ → Pass 4: Coreference resolution (GPT-4o)               │
│ → Entity resolution cascade                              │
│ → MERGE into Neo4j → Embed → Communities                │
└─────────────────────────────────────────────────────────┘
```

A decisão de construir um pipeline custom (versus adotar um framework) é justificada por evidência convergente: (1) extração schema-constrained atinge 98.8% de precisão versus 40% de erro com extração aberta (ODKE+, Plumber); (2) nenhum framework existente modela claims como nós first-class; (3) o custo estimado de $15–45 para 134 episódios torna o projeto viável academicamente.

O stack tecnológico consiste em: Python 3.11+ para toda a lógica; OpenAI API exclusivamente para chamadas LLM e embeddings (eliminando variabilidade entre provedores); Neo4j 5.x como banco de grafos (vector indexes nativos, GDS para Leiden, Cypher para queries); FastMCP para o servidor MCP.

## 4.2 Schema do Knowledge Graph

### 4.2.1 Princípios de Design

O schema segue três princípios derivados da literatura:

**Hierarquia centrada em conversação.** A unidade atômica é um chunk de fala do professor, não um parágrafo de documento. A hierarquia Episode → Segment → Utterance espelha o Episode subgraph do Graphiti (Rasmussen et al., 2025) e o utterance chaining de Di Bratto et al. (2021), preservando a estrutura conversacional que chunking plano destrói.

**Claims como nós first-class (reificação).** Claims, previsões e opiniões são modelados como nós intermediários seguindo o Intermediate Node Pattern do Neo4j, o formalismo I-Node do AIF, e a arquitetura claim-tree do Kialo (18.000+ debates validados em produção). Reificação habilita: metadados arbitrários (confiança, stance, status de verificação), relações claim-a-claim (agreement, contradiction, revision), e provenance rastreável ao momento exato da aula.

**Metadados bi-temporais.** Cada claim carrega duas dimensões temporais: quando foi dita (via Episode publishDate, através da hierarquia de conteúdo) e que período temporal referencia (via propriedades explícitas de data em claims/previsões). Isso segue o modelo de quatro timestamps do Graphiti e o grafo bi-temporal de TG-RAG.

### 4.2.2 Tipos de Nó

A ontologia define 9 tipos de nó em 3 camadas:

**Camada de Conteúdo:**
- **Episode** — uma aula do canal. Propriedades: episodeId, title, publishDate, duration, episodeNumber.
- **Segment** — bloco temático coerente (~10–15 minutos). Propriedades: segmentId, topic, startChar, endChar.
- **Utterance** — chunk de texto (~1500 tokens), unidade para busca vetorial. Propriedades: utteranceId, text, tokenCount, globalIndex, embedding.

**Camada de Entidades:**
- **Person** — figuras históricas, líderes, filósofos. Propriedades: personId, name, aliases, historicalRole, description, embedding.
- **Organization** — impérios, nações, dinastias, religiões, instituições. Propriedades: orgId, name, aliases, orgType, description, embedding.
- **Topic** — conceitos, ideologias, regiões geográficas. Propriedades: topicId, name, category, description, embedding.
- **Event** — guerras, batalhas, tratados, revoluções. Propriedades: eventId, name, eventType, date, description, embedding.

**Camada Analítica:**
- **Claim** — interpretação histórica, previsão geopolítica, ou análise causal feita por Prof. Jiang. Propriedades: claimId, text, claimType, stance, confidence, status, targetDate (para previsões), embedding.
- **Community** — cluster temático detectado por Leiden. Propriedades: communityId, level, name, summary, memberCount, embedding.

### 4.2.3 Tipos de Relação

**Hierarquia de conteúdo:**
- `(:Episode)-[:HAS_SEGMENT {order}]->(:Segment)`
- `(:Segment)-[:HAS_CHUNK {order}]->(:Utterance)`
- `(:Segment)-[:NEXT_SEGMENT]->(:Segment)` — cadeia sequencial
- `(:Utterance)-[:NEXT_CHUNK]->(:Utterance)` — cadeia sequencial

**Dinâmicas históricas:**
- `(:Person)-[:ROLE_IN {title, start_date, end_date}]->(:Organization)`
- `(:Person|:Organization)-[:PARTICIPATED_IN {outcome}]->(:Event)`
- `(:Event)-[:LED_TO|:CAUSED]->(:Event)` — cadeias causais
- `(:Organization)-[:CONTROLLED|:LOCATED_IN]->(:Topic)`

**Grounding de entidades:**
- `(:Utterance)-[:MENTIONS]->(:Person|:Organization|:Topic|:Event)`

**Rede de claims:**
- `(:Claim)-[:EXTRACTED_FROM]->(:Utterance)` — provenance
- `(:Claim)-[:STATED_DURING]->(:Episode)` — âncora temporal
- `(:Claim)-[:ABOUT]->(:Person|:Organization|:Topic|:Event)` — referência
- `(:Claim)-[:SUPPORTS {strength}]->(:Claim)` — reforço
- `(:Claim)-[:CONTRADICTS {strength}]->(:Claim)` — contradição
- `(:Claim)-[:REVISES_PREDICTION]->(:Claim)` — atualização de previsão

**Comunidades:**
- `(:Person|:Topic|:Organization)-[:IN_COMMUNITY]->(:Community)`
- `(:Community)-[:PARENT_COMMUNITY]->(:Community)` — hierarquia Leiden

### 4.2.4 Constraints e Indexes

O banco é configurado com: 9 uniqueness constraints (previnem duplicação por ID); 7 vector indexes (1536 dimensões, cosine similarity, para busca semântica); 2 full-text indexes (busca por keyword em utterances e claims); 5 composite indexes (aceleração de queries frequentes por publishDate, claimType, status, stance, category). Todos utilizam `IF NOT EXISTS` para idempotência.

### 4.2.5 Escala Esperada

Para 134 episódios: ~134 Episode nodes, ~1.300 Segments, ~30.000 Utterances, ~100–300 Persons, ~100 Organizations, ~100–200 Topics, ~200 Events, ~3.000–8.000 Claims, e ~80.000–250.000 relationships. Isso está confortavelmente dentro da capacidade do Neo4j Community Edition.

## 4.3 Pipeline de Preprocessing

### 4.3.1 Download e Transcrição

O áudio é extraído de vídeos do YouTube via yt-dlp (formato mp3, 64kbps — suficiente para fala) e transcrito via Groq Whisper large-v3. Para aulas de ~1 hora que excedem o limite de 25 MB do Groq, o áudio é segmentado em blocos de 20 minutos via ffmpeg, transcritos individualmente, e concatenados. O pipeline é idempotente: se a transcrição já existe, o processamento é pulado.

### 4.3.2 Text Cleaning

Transcrições brutas de áudio contêm artefatos que desperdiçam tokens e confundem a extração:

1. **Normalização Unicode** — smart quotes → standard quotes, encoding consistente
2. **Remoção de artefatos** — `[inaudible]`, `[00:15:32]`, marcações de transcrição
3. **Remoção de fillers** — "uh", "uhm", "you know", "basically" em posição não-semântica
4. **Remoção de stutters** — "the the Roman" → "the Roman"
5. **Normalização de whitespace** — espaços múltiplos, quebras de linha inconsistentes

A limpeza é conservadora: remove apenas ruído óbvio, sem alterar conteúdo semântico ou corrigir erros do transcritor.

### 4.3.3 Two-Pass Chunking

Chunking de tamanho fixo destrói boundaries de segmento — corta ideias ao meio. O sistema emprega uma estratégia de dois passes:

**Level 1 — Segmentação por LLM:** O texto é dividido em sentenças (via regex com proteção de abreviações). Sentenças são agrupadas em janelas deslizantes de 20 sentenças com overlap de 3. Para cada janela, um LLM (GPT-4o-mini, temperatura 0) identifica boundaries de tópico: "onde o professor muda de assunto?". Boundaries são deduplicadas (janelas se sobrepõem) e usadas para construir Segments — blocos de comprimento variável, cada um sobre um tópico coerente.

**Level 2 — Split determinístico com overlap:** Cada Segment é dividido em chunks de no máximo 1500 tokens. A divisão nunca ocorre no meio de uma sentença (sentenças são atômicas). Três sentenças se repetem entre chunks consecutivos (overlap), garantindo que assertivas cross-boundary sejam capturadas em pelo menos um chunk. Chunks com menos de 200 tokens são mesclados ao anterior.

A escolha de 1500 tokens balanceia qualidade de extração (mais contexto por chamada) contra custo (menos tokens por chamada). O overlap de 3 sentenças é baseado no achado de Rook (ISWC 2025) sobre distâncias sujeito-objeto maiores em dados de diálogo.

## 4.4 Pipeline de Extração Multi-Pass

### 4.4.1 Justificativa

Extração single-pass — enviar texto e pedir ao LLM que extraia tudo simultaneamente — é documentadamente inferior. A pesquisa converge: ECR demonstra +16% em F1 de triplas; KGGen (NeurIPS 2025) +18% sobre GraphRAG; SAC-KG (ACL 2024) 89.32% precisão com pipeline de três componentes. Cada pass com tarefa focada permite que o LLM concentre sua capacidade em um problema por vez.

### 4.4.2 Arquitetura dos Passes

| Pass | Tarefa | Modelo | Temperatura | Justificativa |
|------|--------|--------|-------------|---------------|
| 1 | Entity extraction | GPT-4o-mini | 0.0 | Alto volume, structured output, barato |
| 2 | Relationship extraction | GPT-4o-mini | 0.0 | Mesmos chunks, entidades como contexto |
| 3 | Claim/opinion extraction | GPT-4o | 0.0 | Nuance de stance/sentiment requer modelo forte |
| 4 | Coreference resolution | GPT-4o | 0.0 | Raciocínio contextual profundo |
| 5 | Entity resolution | Embeddings + GPT-4o-mini | 0.0 | ~5% dos pares precisam de LLM |

### 4.4.3 Pass 1: Extração de Entidades

**Input:** Um chunk de ~1500 tokens + contexto do episódio (título, número, assuntos conhecidos).
**Output:** Lista de entidades com name, entity_type (person/organization/topic/event), description, aliases, e propriedades type-specific (historicalRole, orgType, topicCategory, eventType).

O prompt instrui o LLM a usar nomes canônicos universalmente reconhecidos ("Alexander the Great", não "Alexander"), incluir todos os aliases mencionados, e classificar cada entidade segundo a ontologia definida. O schema é enforced via OpenAI Structured Outputs — o LLM é fisicamente incapaz de retornar um entity_type fora dos 4 permitidos.

### 4.4.4 Pass 2: Extração de Relações

**Input:** Mesmo chunk + lista de entidades extraídas no Pass 1.
**Output:** Lista de relações com source_entity, target_entity, relationship_type (schema-constrained a 6 tipos), description, e confidence.

As entidades do Pass 1 são injetadas como contexto para que o modelo foque em identificar relações entre entidades conhecidas em vez de re-extraí-las. Os tipos de relação são constrained: ROLE_IN, PARTICIPATED_IN, LED_TO, CAUSED, CONTROLLED, LOCATED_IN.

### 4.4.5 Pass 3: Extração de Claims

**Input:** Mesmo chunk + entidades do Pass 1.
**Output:** Lista de claims com text, claimType, stance, entities_involved, time_reference, confidence, e campos de previsão (targetDate, targetDescription) quando aplicável.

Este é o pass mais crítico — identifica não apenas o que Prof. Jiang menciona, mas o que ele *argumenta*, *interpreta*, *prediz*, e *analisa*. O prompt distingue: factual assertions ("Roma caiu em 476 d.C."), historical interpretations ("A queda de Roma foi causada por desigualdade econômica"), geopolitical predictions ("Os EUA terão uma guerra civil"), e causal analyses ("A logística de Alexandre foi a chave de sua conquista").

GPT-4o é usado aqui (versus mini nos passes anteriores) porque detecção de stance requer compreensão nuançada: a mesma frase pode ser factual ou especulativa dependendo do contexto.

### 4.4.6 Pass 4: Resolução de Correferência

**Input:** Janela deslizante de 4 chunks consecutivos + todas as entidades extraídas nesses chunks.
**Output:** Lista de correferências mapeando referências ambíguas a entidades canônicas (ex: "he" → "Alexander the Great", "the empire" → "Persian Empire").

A janela deslizante é necessária porque pronomes frequentemente referem entidades mencionadas em chunks anteriores. GPT-4o é necessário pelo raciocínio contextual profundo — distinguir "he" referenciando Alexandre vs. Dário depende de compreensão do fluxo argumentativo.

### 4.4.7 Structured Outputs e Caching

Todos os passes usam OpenAI Structured Outputs com schemas Pydantic: a resposta do modelo é garantidamente um JSON válido conforme o schema definido. Isso elimina parsing de texto livre, handling de erros de formato, e retry por output malformado.

Cada chamada LLM é cacheada por hash(system_prompt + user_prompt + model): reprocessar um episódio custa $0 (cache hit). Isso torna o pipeline idempotente — mesmo input produz mesmo output sem custos adicionais — e facilita debugging por inspeção direta das respostas cacheadas.

## 4.5 Entity Resolution

Após extração, a mesma entidade aparece com nomes diferentes entre chunks e episódios. O módulo de entity resolution usa uma cascata que roteia do barato ao caro:

**Estágio 1 — Match exato (40% dos duplicatas):** Normaliza case, whitespace, e acentos. "united states" = "United States" = "UNITED STATES". Custo: zero.

**Estágio 2 — Fuzzy string matching (35%):** RapidFuzz token_sort_ratio > 85 entre entidades do mesmo tipo. Captura "Philip the Second" ≈ "Philip II". Custo: negligível (C-otimizado).

**Estágio 3 — Similaridade de embedding (20%):** Cosine similarity > 0.85 entre embeddings de "name: description". Captura "USA" ≈ "United States" quando descrições se sobrepõem. Custo: uma chamada de embedding por entidade.

**Estágio 4 — Julgamento LLM (5%):** Apenas pares genuinamente ambíguos vão a GPT-4o-mini para decisão merge/no-merge com justificativa.

O princípio guia é conservador: **na dúvida, não mesclar**. Um falso positivo (merge incorreto) corrompe o grafo e é difícil de desfazer. Um falso negativo (manter duplicatas) é inofensivo e corrigível.

O resultado é um merge map: `dict[str, str]` mapeando cada nome de entidade para sua forma canônica, com transitive closure resolvida.

## 4.6 Upsert no Grafo

A inserção segue o padrão MERGE do Neo4j com a regra fundamental: nunca MERGE padrões inteiros com variáveis unbound; sempre vincular nós primeiro, depois MERGE relações entre nós vinculados.

A ordem de inserção garante que endpoints existam antes de criar relações: (1) Episode, (2) Segments + HAS_SEGMENT + NEXT_SEGMENT, (3) Utterances + HAS_CHUNK + NEXT_CHUNK, (4) Entities + MENTIONS, (5) Claims + EXTRACTED_FROM + STATED_DURING + ABOUT, (6) Entity-Entity relationships.

IDs são determinísticos: `make_entity_id("person", "Alexander the Great")` → `"person-alexander-the-great"`. Mesmo input sempre produz mesmo ID, garantindo que MERGE encontre o nó existente em vez de criar duplicata.

Enriquecimento progressivo: `ON MATCH SET` atualiza a descrição se a nova é mais longa (mais informativa), acumula aliases, e incrementa sourceCount — entidades mencionadas em mais chunks recebem descrições progressivamente melhores.

## 4.7 Community Detection e Summarization

Após todas as entidades e relações serem inseridas, o algoritmo de Leiden é executado via Neo4j GDS sobre o subgrafo de entidades co-ocorrentes:

1. **Projeção:** o grafo de entidades (Person, Organization, Topic) com relações ABOUT e entity-entity é projetado in-memory com orientação UNDIRECTED.
2. **Leiden:** executado com `includeIntermediateCommunities: true` para produzir hierarquia, seed aleatória fixa (42) para reprodutibilidade.
3. **Summarization:** para cada comunidade, seus membros e relações internas são coletados e enviados a um LLM que gera um sumário temático descritivo.
4. **Armazenamento:** sumários são salvos em nós Community com embedding para retrieval vetorial.

Para updates incrementais (novo episódio adicionado), `seedProperty` permite que Leiden reutilize a partição anterior como ponto de partida. Apenas comunidades cuja membership mudou precisam de re-sumarização — tipicamente 5–15 comunidades por episódio novo.

## 4.8 Camada de Retrieval

### 4.8.1 Classificação de Query

Um classificador leve (GPT-4o-mini, temperatura 0, structured output) categoriza cada query em 4 tipos:

| Tipo | Exemplo | Estratégia |
|------|---------|-----------|
| Factual | "O que Jiang disse sobre Roma?" | Vector + graph fan-out |
| Comparative | "Compare César e Alexandre" | Cypher template (claims por entidade) |
| Temporal | "Como a previsão sobre os EUA evoluiu?" | Cypher template (claims cronológicas) |
| Thematic | "Quais são os temas dominantes?" | Community summaries |

### 4.8.2 Vector + Graph Fan-Out

O padrão core: busca vetorial encontra chunks semanticamente relevantes; Cypher traversal expande o contexto:

```cypher
CALL db.index.vector.queryNodes('utterance_embeddings', $topK, $queryEmbedding)
YIELD node AS utterance, score
OPTIONAL MATCH (utterance)<-[:HAS_CHUNK]-(seg:Segment)<-[:HAS_SEGMENT]-(ep:Episode)
OPTIONAL MATCH (claim:Claim)-[:EXTRACTED_FROM]->(utterance)
OPTIONAL MATCH (claim)-[:ABOUT]->(entity)
RETURN utterance.text, score, ep.title, ep.publishDate,
       collect(DISTINCT {claim: claim.text, type: claim.claimType, entity: entity.name})
ORDER BY score DESC
```

### 4.8.3 Text-to-Cypher

Para queries que não se encaixam em templates, o sistema gera Cypher a partir de linguagem natural. O prompt inclui o schema completo do grafo, exemplos few-shot, e constraints de read-only. Quando uma query gerada falha, o erro é re-enviado ao LLM para auto-correção — até 2 retries (padrão "self-healing" documentado pela Neo4j).

### 4.8.4 Reciprocal Rank Fusion

Resultados de múltiplas estratégias são fundidos via RRF (Cormack et al., SIGIR 2009):

```
RRF_score(d) = Σ 1/(k + rank(d))  onde k=60
```

Isso combina rankings de diferentes estratégias sem necessitar calibração de scores — ideal quando vector scores e Cypher results não são diretamente comparáveis.

### 4.8.5 Context Assembly

O contexto final separa passages textuais de informação estruturada do grafo, permitindo que o LLM client-side distinga entre "o que foi dito literalmente" e "o que o grafo sabe estruturalmente".

## 4.9 MCP Server

O servidor MCP expõe 6 tools:

1. **search_lectures** — busca semântica + fan-out para queries gerais
2. **find_claims** — busca de claims com filtros (tipo, entidade, stance)
3. **compare_entities** — claims de múltiplas entidades side-by-side
4. **topic_timeline** — evolução cronológica de claims sobre um tópico
5. **run_cypher** — natural language → Cypher para queries analíticas ad-hoc
6. **get_schema** — retorna o schema do grafo para self-discovery

Princípios de design: domain tools para queries comuns (otimizados, rápidos, fáceis de usar corretamente); generic tools para exploração (flexibilidade para queries imprevistas); resultados capped (previne overflow de contexto); dados estruturados retornados (o LLM client gera texto final).

## 4.10 Custo e Otimizações

Três mecanismos reduzem custos:

**Prompt caching (50% desconto):** O system prompt + schema é idêntico entre chunks de um episódio. Processamento sequencial mantém o prefixo no cache do provider, reduzindo custo de tokens de input pela metade.

**Tiered model usage:** GPT-4o-mini (80% das chamadas — passes 1, 2, 5) custa ~10× menos que GPT-4o (20% — passes 3, 4). Tarefas simples no modelo barato, tarefas nuançadas no modelo capaz.

**Caching local:** Hash de cada chamada armazenado em disco. Reprocessamento custa $0.

Estimativa por episódio (~1 hora, ~17K tokens de transcrição): Passes 1–2 (GPT-4o-mini) ~$0.02; Pass 3 (GPT-4o) ~$0.04; Pass 4 (GPT-4o) ~$0.02; Embeddings ~$0.01; Community summarization (amortizado) ~$0.01. **Total por episódio: ~$0.10. Para 134 episódios: ~$13–15.** Ordens de magnitude abaixo dos $33K estimados para Microsoft GraphRAG em escala similar.
