# Design da Camada de Retrieval

## O que é

A camada de retrieval é o que transforma uma pergunta do usuário em contexto relevante extraído do knowledge graph. É a ponte entre "o que o usuário quer saber" e "o que o grafo contém".

## Arquitetura: Query-Adaptive Retrieval

Nem toda pergunta deve ser respondida da mesma forma. A pesquisa (Adaptive-RAG, NAACL 2024) demonstra que classificar a query e rotear para a estratégia ótima supera abordagens one-size-fits-all.

Nosso sistema classifica queries em 4 tipos e roteia cada um para uma estratégia diferente:

### Os 4 tipos de query

| Tipo | Exemplo | Estratégia |
|------|---------|-----------|
| **Factual** | "What did Jiang say about the fall of Rome?" | Vector search + graph fan-out |
| **Comparative** | "How do Caesar and Alexander differ as military leaders?" | Cypher template (claims by entity) |
| **Temporal** | "How has Jiang's prediction about the US evolved?" | Cypher template (chronological claims) |
| **Thematic** | "What are the main themes across all episodes?" | Community summaries |

### Por que 4 estratégias (não uma só)

- **Vector search** é ótimo para "encontrar passagens relevantes" mas não sabe comparar entidades
- **Cypher templates** são ótimos para queries estruturadas mas não encontram passagens semanticamente similares
- **Text-to-Cypher** é flexível mas menos confiável (pode gerar Cypher inválido)
- **Community summaries** respondem perguntas globais que nenhuma passagem individual responde

Combinar todas via RRF (Reciprocal Rank Fusion) dá o melhor dos mundos.

## Componentes

### 1. Query Classifier (`classifier.py`)

Um LLM leve (GPT-4o-mini) que recebe a query do usuário e retorna:
- O tipo (factual/comparative/temporal/thematic)
- Entidades mencionadas (para filtrar)
- Time range (se houver)
- Tópicos mencionados

### 2. Vector Search + Graph Fan-Out (`vector_search.py`)

O padrão core:
1. Embed a query do usuário
2. Busca os top-K chunks mais similares (vector index)
3. Para cada chunk encontrado, faz graph traversal:
   - Qual episódio? Qual segmento?
   - Quais entidades são mencionadas?
   - Quais claims foram extraídas desse chunk?
   - Quais tópicos essas claims referenciam?

Isso retorna não apenas o texto relevante, mas todo o contexto estruturado ao redor.

### 3. Text-to-Cypher (`text2cypher.py`)

Para queries analíticas que mapeiam naturalmente para patterns de grafo:
1. Passa o schema do grafo + few-shot examples + a query ao LLM
2. LLM gera uma query Cypher
3. Executa a query
4. Se falhar: alimenta o erro de volta ao LLM (self-healing, até 2 retries)

Exemplos de queries que Text-to-Cypher resolve bem:
- "How many episodes discuss the Roman Empire?"
- "Which topics have the most claims?"
- "What predictions were verified as correct?"

### 4. Context Assembler + RRF (`assembler.py`)

Combina resultados de todas as estratégias ativas:
1. Cada estratégia retorna uma lista rankeada de resultados
2. RRF funde as listas: score(d) = Σ 1/(k + rank_i(d)), k=60
3. O contexto final é montado separando:
   - Passagens relevantes (texto dos chunks)
   - Contexto do grafo (claims, entidades, relações)
   - Resultados estruturados (de Text-to-Cypher)

## Fluxo completo

```
User query
    → Classifier (tipo + entidades + time range)
    → Strategy Router:
        ├── Factual → Vector + Graph Fan-Out
        ├── Comparative → Cypher Template (claims by entity)
        ├── Temporal → Cypher Template (chronological)
        └── Thematic → Community Summary Search
    → Results from each active strategy
    → RRF Fusion (merge ranked lists)
    → Context Assembly (format for LLM)
    → Final context string (ready for generation)
```

## Notas para o TCC

### O que documentar no artigo:
1. **Query-adaptive retrieval** — por que é superior a one-size-fits-all (Adaptive-RAG)
2. **Vector + Graph fan-out** — o padrão dual-path (semântico + estrutural)
3. **Text-to-Cypher com self-healing** — flexibilidade + robustez
4. **RRF** — como fundir resultados de múltiplas estratégias (Cormack et al., SIGIR 2009)
5. **Community summaries para queries globais** — o que Microsoft GraphRAG faz bem
6. **Comparação com RAG tradicional** — o que nosso sistema responde que flat RAG não consegue
