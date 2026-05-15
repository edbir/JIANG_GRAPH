# Neo4j Client — Documentação Técnica

## O que é

`src/graph/neo4j_client.py` é o módulo que gerencia a conexão com o Neo4j e configura o schema do banco (constraints e indexes). É a fundação sobre a qual o upsert e o retrieval operam.

## Responsabilidades

| Faz | Não faz |
|-----|---------|
| Conectar ao Neo4j | Inserir dados (→ upsert.py) |
| Criar constraints | Rodar community detection (→ communities.py) |
| Criar indexes (vector, fulltext, composite) | Fazer queries de retrieval (→ retrieval/) |
| Executar Cypher genérico | Decidir o que inserir |
| Verificar conectividade | Gerenciar transações complexas |

## Como usar

```python
from src.graph.neo4j_client import GraphClient

# Como context manager (recomendado)
async with GraphClient() as client:
    # Setup inicial (rodar uma vez)
    await client.setup_schema()
    
    # Executar queries
    results = await client.run(
        "MATCH (e:Episode) WHERE e.episodeNumber = $num RETURN e.title",
        {"num": 42}
    )
    
    # Monitoramento
    counts = await client.get_node_counts()
    print(counts)  # {"Episode": 5, "Claim": 342, ...}
```

## Schema criado por setup_schema()

### Constraints (unicidade)

Cada tipo de nó tem um campo ID único. Se tentarmos criar um nó com ID duplicado, Neo4j rejeita.

| Label | Campo único |
|-------|-------------|
| Episode | episodeId |
| Segment | segmentId |
| Utterance | utteranceId |
| Person | personId |
| Organization | orgId |
| Topic | topicId |
| Event | eventId |
| Claim | claimId |
| Community | communityId |

### Vector Indexes

Permitem busca por similaridade semântica (approximate nearest neighbor).

| Index | Nó | Campo | Dimensões | Similaridade |
|-------|-----|-------|-----------|-------------|
| utterance_embeddings | Utterance | embedding | 1536 | cosine |
| claim_embeddings | Claim | embedding | 1536 | cosine |
| person_embeddings | Person | embedding | 1536 | cosine |
| org_embeddings | Organization | embedding | 1536 | cosine |
| topic_embeddings | Topic | embedding | 1536 | cosine |
| event_embeddings | Event | embedding | 1536 | cosine |
| community_embeddings | Community | embedding | 1536 | cosine |

### Full-Text Indexes

Permitem busca por keyword (Lucene-based).

| Index | Nó | Campo |
|-------|-----|-------|
| utterance_text | Utterance | text |
| claim_text | Claim | text |

### Composite Indexes

Aceleram filtros comuns em queries.

| Index | Nó | Campo | Uso típico |
|-------|-----|-------|-----------|
| episode_date | Episode | publishDate | Queries temporais |
| claim_type | Claim | claimType | Filtrar por tipo de claim |
| claim_status | Claim | status | Filtrar predictions verificadas |
| claim_stance | Claim | stance | Filtrar por posição do Jiang |
| topic_category | Topic | category | Filtrar tópicos por categoria |

## Conexão

O client lê credenciais do `config.py` (que lê do `.env`):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme
```

O Neo4j roda via Docker (ver `docker-compose.yml` na raiz).

## Notas de implementação

- **Async:** Todo o client é async (usa `neo4j.AsyncGraphDatabase`). Isso é consistente com o resto do pipeline (extractor, chunker).
- **Idempotente:** `setup_schema()` usa `IF NOT EXISTS` em tudo. Pode rodar quantas vezes quiser.
- **Connection pool:** O driver Neo4j gerencia um pool de conexões internamente. Não precisamos nos preocupar com isso.
- **APOC:** Os métodos `get_node_counts()` e `get_relationship_counts()` usam APOC (um plugin de utilidades do Neo4j). Se APOC não estiver instalado, esses métodos falham — mas não são críticos para o pipeline.
