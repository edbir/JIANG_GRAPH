# Retrieval Layer — Documentação Técnica

## O que é

A camada de retrieval transforma uma pergunta do usuário em contexto relevante extraído do knowledge graph. É composta por 4 módulos que trabalham juntos.

## Posição no pipeline

```
User query → [retrieval/] → assembled context → MCP server / generation
```

## Os 4 módulos

### 1. `classifier.py` — Query Classification

Classifica a query em um de 4 tipos usando GPT-4o-mini com structured output:

| Tipo | Exemplo | Estratégia ativada |
|------|---------|-------------------|
| factual | "What did Jiang say about Rome?" | Vector + graph fan-out |
| comparative | "How do Caesar and Alexander differ?" | Claim search + Text-to-Cypher |
| temporal | "How has the prediction about the US evolved?" | Chronological claims + T2C |
| thematic | "What are the main themes?" | Community summaries |

Também extrai: entidades mencionadas, time range, tópicos.

### 2. `vector_search.py` — Vector Search + Graph Fan-Out

O padrão core de retrieval:

```
Query embedding → Vector index (top-K chunks) → Graph traversal:
    → Episode (title, date)
    → Segment (topic)
    → Entities (MENTIONS)
    → Claims (EXTRACTED_FROM)
    → Topics (ABOUT)
```

Também oferece:
- `search_claims()` — busca direta em embeddings de claims
- `search_communities()` — busca em embeddings de community summaries

### 3. `text2cypher.py` — Natural Language → Cypher

Para queries analíticas/estruturais:
1. LLM gera Cypher a partir da pergunta + schema do grafo + few-shot examples
2. Executa contra Neo4j
3. Se falhar: self-healing (erro → LLM → query corrigida, até 2 retries)

Safety: NUNCA executa write queries (regex check antes de executar).

### 4. `assembler.py` — Context Assembly + RRF

Combina resultados de todas as estratégias:
1. **RRF (Reciprocal Rank Fusion):** Funde listas rankeadas sem normalização de scores
2. **Deduplicação:** Remove resultados repetidos
3. **Formatação:** Monta contexto estruturado em seções:
   - `## Relevant Passages` (texto dos chunks)
   - `## Knowledge Graph Context` (claims estruturadas)
   - `## Structured Query Results` (de Text-to-Cypher)
   - `## Thematic Context` (community summaries)

## Entry point

```python
from src.retrieval.assembler import retrieve

context = await retrieve(graph_client, openai_client, "What did Jiang say about Rome?")
# Returns formatted context string ready for LLM generation
```

## Fluxo completo

```
User query
    → classify_query() → QueryClassification(type=factual, entities=["Rome"])
    → Strategy Router:
        factual → vector_search_with_fanout(entity_filter="Rome")
    → Results: [RetrievedPassage, RetrievedPassage, ...]
    → assemble_context(passages=results)
    → "## Relevant Passages\n[Episode 5, 2024-01-15]: The Roman Empire fell..."
```

## RRF Formula

```
RRF_score(d) = Σ 1/(k + rank_i(d))    where k=60
```

Documento rankeado #1 em duas listas: 1/61 + 1/61 = 0.033
Documento rankeado #1 em uma e #10 em outra: 1/61 + 1/70 = 0.031

Favorece documentos que aparecem em MÚLTIPLAS estratégias (consenso = relevância).

## Referências

- Adaptive-RAG (Jeong et al., NAACL 2024) — query-adaptive retrieval
- RRF (Cormack et al., SIGIR 2009) — fusion formula
- RAG-Fusion (Rackauckas, 2024) — +8-10% accuracy com RRF
- Neo4j Text2Cypher 2024 dataset — 44,387 instances
