# Communities — Documentação Técnica

## O que é

`src/graph/communities.py` detecta clusters temáticos no grafo usando o algoritmo de Leiden e gera summaries legíveis via LLM para cada cluster.

## Posição no pipeline

```
upsert.py (grafo populado) → [communities.py] → Community nodes com summaries
                                                        ↓
                                              retrieval/ (queries temáticas)
```

## O que são communities

Grupos de entidades densamente conectadas que representam TEMAS:
- "Guerras Púnicas": Roma, Cartago, Aníbal, Batalha de Canas...
- "Conquistas de Alexandre": Alexandre, Macedônia, Pérsia, Gaugamela...

## O processo (4 etapas)

### 1. Projeção do grafo (GDS)
```
Projeta: Person, Organization, Topic, Event + suas relações
Não projeta: Episode, Segment, Utterance (são estruturais, não temáticos)
Orientação: UNDIRECTED (para clustering, direção não importa)
```

### 2. Leiden algorithm
```
Input: Grafo projetado in-memory
Output: communityId atribuído a cada entidade
Hierarquia: Múltiplos níveis (fine → broad)
Parâmetro gamma: controla resolução (mais alto = mais comunidades menores)
```

### 3. Summarization (LLM)
```
Para cada comunidade:
  1. Coleta membros + relações internas
  2. Formata como texto
  3. GPT-4o-mini gera summary de 2-4 frases
  4. Armazena summary + embedding no nó Community
```

### 4. Cleanup
```
Drop da projeção in-memory (libera RAM do GDS)
```

## Funções principais

| Função | O que faz |
|--------|-----------|
| `project_entity_graph()` | Projeta entidades no GDS |
| `run_leiden()` | Roda Leiden, escreve communityId nos nós |
| `get_community_members()` | Coleta membros + relações de uma comunidade |
| `summarize_community()` | Gera summary LLM para uma comunidade |
| `create_community_nodes()` | Cria nós Community com summaries + embeddings |
| `detect_and_summarize_communities()` | **Entry point:** orquestra tudo |

## Como usar

```python
from src.graph.neo4j_client import GraphClient
from src.graph.communities import detect_and_summarize_communities

async with GraphClient() as client:
    stats = await detect_and_summarize_communities(
        client=client,
        llm_client=openai_client,
        use_seed=False,  # True para updates incrementais
        gamma=1.0,
    )
    print(stats)
    # {"projected": {...}, "leiden": {"communityCount": 23, ...}, "communities_created": 20}
```

## Incrementalidade

Para novos episódios:
```python
# use_seed=True reutiliza comunidades anteriores como ponto de partida
stats = await detect_and_summarize_communities(client, llm_client, use_seed=True)
```

Leiden com seed só reprocessa entidades afetadas. Apenas comunidades que mudaram precisam de re-summarization.

## Dependências

- **Neo4j GDS plugin:** Obrigatório. Incluído no docker-compose.yml.
- **OpenAI API:** Para summarization e embeddings.

## Filtros

- Comunidades com < 3 membros são ignoradas (provavelmente ruído)
- Limite de 50 membros e 100 relações por community no prompt (evitar overflow)

## Nó Community no grafo

```
(:Community {
    communityId: "community-0-42",
    level: 0,
    summary: "This community represents...",
    memberCount: 12,
    leidenId: 42,
    embedding: [0.012, -0.034, ...]  // 1536 dims
})
```

Conectado via: `(:Entity)-[:IN_COMMUNITY]->(:Community)`
