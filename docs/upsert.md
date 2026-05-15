# Upsert — Documentação Técnica

## O que é

`src/graph/upsert.py` insere dados extraídos no Neo4j de forma idempotente (rodar duas vezes = mesmo resultado, sem duplicatas).

## Posição no pipeline

```
extractor.py → extraction results
resolver.py  → merge map            → [upsert.py] → Neo4j graph
chunker.py   → segments + chunks
```

## A regra de ouro

```cypher
-- ❌ NUNCA: MERGE pattern inteiro com variáveis não-bound
MERGE (a:Person {name: "X"})-[:KNOWS]->(b:Person {name: "Y"})

-- ✅ SEMPRE: bind nós primeiro, depois MERGE relação
MATCH (a:Person {personId: $id1})
MATCH (b:Person {personId: $id2})
MERGE (a)-[:KNOWS]->(b)
```

## Ordem de inserção

1. Episode
2. Segments (→ Episode)
3. Utterances/Chunks (→ Segments)
4. Entities (Person, Org, Topic, Event)
5. MENTIONS relationships (Utterance → Entity)
6. Claims (→ Utterance, → Episode, → Entities)
7. Entity↔Entity relationships (ROLE_IN, LED_TO, etc.)

## Estratégia de IDs

| Nó | Formato | Exemplo |
|----|---------|---------|
| Episode | `ep{N:03d}` | `ep001` |
| Segment | `{epId}-seg{N:02d}` | `ep001-seg03` |
| Utterance | `{epId}-seg{N}-chunk{M}` | `ep001-seg03-chunk02` |
| Person | `person-{normalized}` | `person-alexander-the-great` |
| Organization | `org-{normalized}` | `org-roman-empire` |
| Topic | `topic-{normalized}` | `topic-democracy` |
| Event | `event-{normalized}` | `event-battle-of-thermopylae` |
| Claim | `claim-{hash[:12]}` | `claim-a3f2b9c1d4e5` |

## Enriquecimento progressivo

Quando uma entidade é encontrada novamente:
- Descrição: mantém a MAIS LONGA
- Aliases: acumula (union)
- sourceCount: incrementa (+1)

## Funções principais

| Função | O que insere |
|--------|-------------|
| `upsert_episode()` | Nó Episode |
| `upsert_segments()` | Nós Segment + HAS_SEGMENT + NEXT_SEGMENT |
| `upsert_utterances()` | Nós Utterance + HAS_CHUNK + NEXT_CHUNK |
| `upsert_entity()` | Nó Entity + MENTIONS |
| `upsert_claim()` | Nó Claim + EXTRACTED_FROM + STATED_DURING + ABOUT |
| `upsert_relationship()` | Relação Entity↔Entity |
| `store_embedding()` | Vetor de embedding em nó existente |
| `upsert_episode_data()` | **Entry point:** orquestra tudo para um episódio |

## Como usar

```python
from src.graph.neo4j_client import GraphClient
from src.graph.upsert import upsert_episode_data

async with GraphClient() as client:
    await client.setup_schema()  # uma vez
    
    counts = await upsert_episode_data(
        client=client,
        episode_id="ep001",
        episode_title="The Fall of Rome",
        episode_number=1,
        segments=segments,
        chunks=chunks,
        extraction_results=results,  # output de extractor.process_episode()
        merge_map=merge_map,         # output de resolver.resolve_entities()
    )
    print(counts)  # {"entities": 45, "relationships": 23, "claims": 18, ...}
```

## Confidence threshold

Entidades, relações, e claims com `confidence < 0.7` NÃO são inseridas no grafo. Isso filtra extrações de baixa qualidade antes que contaminem o knowledge graph.
