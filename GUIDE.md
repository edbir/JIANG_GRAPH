# Guia Completo do Projeto — Para Humanos e IAs

## Quem deve ler isso

- **Eduardo** voltando ao projeto depois de dias/semanas sem mexer
- **Uma IA (Kiro ou outro)** começando a trabalhar neste repositório pela primeira vez
- **Qualquer pessoa** que queira entender o sistema sem ler todo o código

## O que é este projeto (em 30 segundos)

Um sistema que pega aulas de história do YouTube (Prof. Jiang, canal "Predictive History"), transcreve, extrai conhecimento estruturado (entidades, relações, claims, previsões), armazena em um grafo Neo4j, e permite que LLMs consultem esse grafo via MCP.

É o TCC do Eduardo em Ciência da Computação na UFRGS.

---

## Mapa Mental do Sistema

```
🎬 YouTube (Predictive History)
    │
    ▼ (download_and_transcribe.py: yt-dlp baixa áudio)
🔊 Áudio mp3 (64kbps)
    │
    ▼ (download_and_transcribe.py: Groq Whisper transcreve)
📝 Transcrição bruta
    │
    ▼ (text_cleaner.py remove ruído)
📝 Texto limpo
    │
    ▼ (chunker.py: LLM detecta tópicos + split com overlap)
📦 Chunks (~1500 tokens cada)
    │
    ▼ (extractor.py: 5 passes com OpenAI)
🧠 Conhecimento extraído:
    ├── Entidades (Alexandre, Roma, Batalha X...)
    ├── Relações (Alexandre PARTICIPATED_IN Batalha X)
    ├── Claims ("Roma caiu por desigualdade econômica")
    └── Correferências ("he" = Alexandre)
    │
    ▼ (resolver.py: deduplicação)
🔗 Entidades únicas (merge map)
    │
    ▼ (upsert.py: MERGE no Neo4j)
🕸️ Knowledge Graph (Neo4j)
    │
    ▼ (communities.py: Leiden + LLM summaries)
🏘️ Comunidades temáticas
    │
    ▼ (retrieval/: classify → route → search → fuse)
📋 Contexto estruturado
    │
    ▼ (mcp/server.py: 6 tools via MCP)
🤖 Qualquer LLM pode consultar
```

---

## Os 5 Conceitos que Você PRECISA Entender

### 1. Por que grafo e não só vetores?

RAG tradicional: "encontre os 5 chunks mais parecidos com a pergunta".
Nosso sistema: "encontre os chunks + quem é mencionado + que claims foram feitas + como se conectam + em que episódio + que comunidade temática pertencem".

O grafo preserva ESTRUTURA. Vetores preservam SEMÂNTICA. Combinamos os dois.

### 2. Por que multi-pass (não single-pass)?

Se você pede ao LLM "extraia entidades, relações, claims, e resolva correferências" tudo de uma vez, ele faz tudo mal. Se você pede uma coisa por vez, ele faz cada uma bem.

Pesquisa confirma: +16% a +18% de melhoria com multi-pass.

### 3. Por que claims são NÓS (não propriedades)?

Se "Roma caiu por desigualdade" fosse uma propriedade de uma aresta Roma→Queda, não poderíamos:
- Dizer que essa claim CONTRADIZ outra claim
- Rastrear em qual episódio foi dita
- Atribuir confiança
- Marcar como "verificada" ou "refutada"

Como NÓ, a claim pode ter relações próprias, propriedades, e participar da estrutura do grafo.

### 4. Por que 4 tipos de query?

- "O que Jiang disse sobre Roma?" → busca semântica (vetores)
- "Compare César e Alexandre" → busca estrutural (claims por entidade)
- "Como a previsão sobre os EUA evoluiu?" → busca temporal (claims ordenadas por data)
- "Quais são os temas gerais?" → busca temática (community summaries)

Uma estratégia não serve para todas. Classificamos e roteamos.

### 5. Por que MCP?

MCP é um protocolo que permite qualquer LLM (Claude, GPT, etc.) chamar funções do nosso sistema. Em vez de construir uma UI, construímos uma API que LLMs sabem usar. O LLM vira a interface do usuário.

---

## Como Rodar o Sistema (passo a passo)

### Pré-requisitos
```bash
# 1. Ter Docker instalado
# 2. Ter Python 3.11+
# 3. Ter chaves de API (OpenAI + Groq)
# 4. Ter ffmpeg instalado (sudo apt install ffmpeg)
```

### Setup inicial (uma vez)
```bash
# Subir Neo4j
docker compose up -d

# Instalar dependências Python
pip install -r requirements.txt

# Configurar chaves
cp .env.example .env
# Editar .env com suas chaves (OPENAI_API_KEY + GROQ_API_KEY)
```

### Baixar e transcrever episódios
```bash
# Opção 1: Um vídeo por vez
python -m scripts.download_and_transcribe \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --episode-id ep001

# Opção 2: Batch (vários episódios de uma vez)
# Crie um arquivo episodes.txt com formato:
#   ep001 https://www.youtube.com/watch?v=ABC123 The Fall of Rome
#   ep002 https://www.youtube.com/watch?v=DEF456 Alexander's Conquests
python -m scripts.download_and_transcribe --batch episodes.txt
```

### Ingerir um episódio
```bash
# 1. Ter transcrição em data/transcripts/ep001.txt (passo anterior)

# 2. Rodar o pipeline completo
python -m scripts.ingest_episode \
    --transcript data/transcripts/ep001.txt \
    --episode-id ep001 \
    --title "The Fall of the Roman Republic" \
    --number 1 \
    --communities
```

### Consultar o grafo
```bash
# Iniciar o MCP server
python -m src.mcp.server

# Ou abrir Neo4j Browser: http://localhost:7474
```

---

## Mapa de Arquivos (o que cada um faz)

### Código (`src/`)

| Arquivo | Uma frase |
|---------|-----------|
| `config.py` | Todas as configurações em um lugar (lê do .env) |
| `preprocessing/text_cleaner.py` | Remove "uh", "[inaudible]", stutters do texto |
| `preprocessing/chunker.py` | Divide texto em pedaços inteligentes (LLM + overlap) |
| `extraction/schemas.py` | Define O QUE cada pass de extração produz (Pydantic) |
| `extraction/prompts.py` | Define COMO pedir ao LLM (system + user prompts) |
| `extraction/extractor.py` | Orquestra os 5 passes, chama a API, cacheia respostas |
| `extraction/resolver.py` | Deduplica entidades (exact → fuzzy → embedding → LLM) |
| `graph/neo4j_client.py` | Conecta ao Neo4j, cria constraints e indexes |
| `graph/upsert.py` | Insere dados no grafo (MERGE idempotente) |
| `graph/communities.py` | Detecta temas (Leiden) e gera summaries (LLM) |
| `retrieval/classifier.py` | Classifica query em 4 tipos |
| `retrieval/vector_search.py` | Busca semântica + expansão no grafo |
| `retrieval/text2cypher.py` | Pergunta → Cypher → resultado (com self-healing) |
| `retrieval/assembler.py` | Funde resultados (RRF) e monta contexto final |
| `mcp/server.py` | Expõe 6 tools para LLMs via MCP |

### Scripts (`scripts/`)

| Arquivo | Uma frase |
|---------|-----------|
| `download_and_transcribe.py` | YouTube URL → áudio (yt-dlp) → transcript.txt (Groq Whisper) |
| `ingest_episode.py` | Pipeline completo: transcript → graph (o que você roda depois) |
| `validate_graph.py` | QA: detecta orphans, duplicatas, embeddings faltando, etc. |

### Documentação (`docs/`)

| Pasta/Arquivo | Uma frase |
|---------------|-----------|
| `PROJECT_OVERVIEW.md` | Visão geral completa (leia PRIMEIRO) |
| `*.md` (raiz) | Referência técnica de cada módulo |
| `design/*.md` | Design thinking (o PORQUÊ de cada decisão) |

### Preparação do TCC (`preparacao/`)

| Arquivo | Uma frase |
|---------|-----------|
| `01-07` | Escopo + sugestões de estudo para cada seção do artigo |

---

## Fluxo de Dados Detalhado (para quem quer entender a fundo)

### Fase 0: Download e Transcrição

```
Input: YouTube URL (e.g., "https://www.youtube.com/watch?v=VIDEO_ID")

download_and_transcribe.py:
  → download_audio()         : yt-dlp extrai áudio como mp3 (64kbps)
  → split_audio()            : se > 20 min, divide em segmentos (ffmpeg)
  → transcribe_full_audio()  : cada segmento → Groq Whisper large-v3
  → concatena segmentos      : junta todos os textos em ordem

Output: data/transcripts/ep001.txt (~15,000-20,000 tokens de texto bruto)
```

### Fase 1: Preprocessing

```
Input: "So, uh, Alexander the Great, you know, he conquered the the Persian Empire..."

text_cleaner.py:
  → normalize_unicode()     : smart quotes → standard
  → remove_artifacts()      : [inaudible], timestamps → gone
  → remove_fillers()        : "uh", "you know" → gone
  → remove_stutters()       : "the the" → "the"
  → normalize_whitespace()  : double spaces → single

Output: "Alexander the Great conquered the Persian Empire..."
```

### Fase 2: Chunking

```
Input: Texto limpo (~15,000 tokens)

chunker.py Level 1 (LLM):
  → split_into_sentences()           : texto → ~500 sentenças
  → prepare_segmentation_windows()   : sentenças → janelas de 20
  → segment_with_llm()               : LLM identifica mudanças de tópico
  → build_segments_from_boundaries() : índices → Segment objects

  Output: [Segment("Alexander's conquests", sentences[0:45]),
           Segment("The Persian Empire's fall", sentences[45:82]),
           ...]

chunker.py Level 2 (determinístico):
  → chunk_segment()  : cada Segment → chunks de ≤1500 tokens
                       com 3 sentenças de overlap

  Output: [Chunk("ep001-seg00-chunk00", text="...", tokens=1423),
           Chunk("ep001-seg00-chunk01", text="...", tokens=1389),
           ...]
```

### Fase 3: Extração

```
Input: Um chunk de ~1500 tokens

extractor.py Pass 1 (GPT-4o-mini):
  "Que entidades existem neste texto?"
  → [Entity("Alexander the Great", person, military_leader),
     Entity("Persian Empire", organization, empire),
     Entity("Battle of Gaugamela", event, battle)]

extractor.py Pass 2 (GPT-4o-mini):
  "Que relações existem entre essas entidades?"
  → [Relationship("Alexander" → PARTICIPATED_IN → "Battle of Gaugamela"),
     Relationship("Battle of Gaugamela" → LED_TO → "Fall of Persian Empire")]

extractor.py Pass 3 (GPT-4o):
  "Que claims/interpretações Jiang faz?"
  → [Claim("Alexander's success was due to Philip's military reforms",
           type=causal_analysis, stance=neutral)]

extractor.py Pass 4 (GPT-4o, janela de 4 chunks):
  "Quem é 'he'? Quem é 'the empire'?"
  → [Coreference("he" → "Alexander the Great", chunk 1),
     Coreference("the empire" → "Persian Empire", chunk 2)]
```

### Fase 4: Resolution + Upsert

```
resolver.py:
  "Alexander" e "Alexander the Great" são a mesma pessoa?
  → Stage 1 (exact): "alexander" = "alexander the great"? Não.
  → Stage 2 (fuzzy): token_sort_ratio = 72. Abaixo de 85. Não.
  → Stage 3 (embedding): cosine("Alexander: Macedonian king", 
                                 "Alexander the Great: conqueror") = 0.91. SIM!
  → merge_map: {"Alexander": "Alexander the Great"}

upsert.py:
  → MERGE (ep:Episode {episodeId: "ep001"})
  → MERGE (seg:Segment {segmentId: "ep001-seg00"})
  → MERGE (u:Utterance {utteranceId: "ep001-seg00-chunk00"})
  → MERGE (p:Person {personId: "person-alexander-the-great"})
  → MERGE (u)-[:MENTIONS]->(p)
  → MERGE (c:Claim {claimId: "claim-a3f2b9..."})
  → MERGE (c)-[:EXTRACTED_FROM]->(u)
  → MERGE (c)-[:ABOUT]->(p)
```

### Fase 5: Communities + Retrieval

```
communities.py:
  → project_entity_graph()  : projeta entidades no GDS
  → run_leiden()            : detecta clusters
  → summarize_community()  : LLM gera "This community represents Alexander's
                              conquest of Persia and the spread of Hellenistic culture"

retrieval (quando usuário pergunta):
  User: "What did Jiang say about Alexander?"
  → classifier: type=factual, entities=["Alexander the Great"]
  → vector_search: top-5 chunks mais similares + graph fan-out
  → assembler: monta contexto com passages + claims + entities
  → MCP server retorna ao LLM client
  → LLM client gera resposta em linguagem natural
```

---

## Decisões-Chave (para não esquecer)

| Decisão | Escolha | Por quê |
|---------|---------|---------|
| LLM provider | OpenAI exclusivo | Simplicidade, sem variabilidade |
| Chunking Level 1 | LLM-based | Superior a regex/embeddings |
| Extração | 5 passes | +16-18% sobre single-pass |
| Claims | Nós (reificação) | Permite metadados e relações claim↔claim |
| Entity resolution | Cascata 4 estágios | 95% sem LLM, conservador |
| Retrieval | 4 estratégias por tipo | Cada query tem abordagem ótima |
| Interface | MCP | Qualquer LLM pode consultar |
| Temporal | Mantido | Jiang faz previsões geopolíticas verificáveis |
| Speaker attribution | Removido | Jiang é o único speaker |

---

## Quando Algo Dá Errado

| Sintoma | Provável causa | Onde olhar |
|---------|---------------|-----------|
| Extração retorna vazio | Chunk muito curto ou sem conteúdo | `text_cleaner.py` (removeu demais?) |
| Entidades duplicadas no grafo | Entity resolution falhou | `resolver.py` (thresholds) |
| Claims genéricas demais | Prompt de extração fraco | `prompts.py` Pass 3 |
| Retrieval não encontra nada | Embeddings não gerados | `ingest_episode.py` step 7 |
| Text-to-Cypher falha sempre | Schema no prompt desatualizado | `text2cypher.py` GRAPH_SCHEMA |
| Neo4j não conecta | Docker não está rodando | `docker compose up -d` |
| Custo alto demais | Sem prompt caching | Verificar se system prompt é estático |

---

## Próximos Passos (quando tiver API + Neo4j)

1. ✅ Configurar `.env` com chaves (OPENAI_API_KEY + GROQ_API_KEY)
2. ✅ `docker compose up -d`
3. ✅ `pip install -r requirements.txt`
4. ✅ Instalar ffmpeg (`sudo apt install ffmpeg`)
5. Baixar e transcrever 5 episódios (`download_and_transcribe.py`)
6. Rodar `ingest_episode.py` para cada um
7. Testar queries de `tests/test_queries.md`
8. Iterar nos prompts baseado nos resultados
9. Documentar resultados no TCC.md
10. Escrever o artigo seção por seção
