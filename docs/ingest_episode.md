# Ingest Episode — Documentação Técnica

## O que é

`scripts/ingest_episode.py` é o script que orquestra o pipeline de ingestion completo para um episódio. É o primeiro script que você roda quando tem uma transcrição pronta.

## O que faz (8 etapas)

```
1. Lê o arquivo de transcrição
2. Limpa o texto (text_cleaner)
3. Chunka (chunker: LLM segmentation + token split)
4. Extrai conhecimento (extractor: 5 passes)
5. Resolve entidades (resolver: deduplicação)
6. Insere no Neo4j (upsert: MERGE idempotente)
7. Gera embeddings (OpenAI text-embedding-3-small)
8. (Opcional) Roda community detection (Leiden + summarization)
```

## Como usar

### Via linha de comando:
```bash
python -m scripts.ingest_episode \
    --transcript data/transcripts/ep001.txt \
    --episode-id ep001 \
    --title "The Fall of the Roman Republic" \
    --number 1 \
    --context "Roman politics, Caesar, civil war" \
    --communities  # opcional: re-roda Leiden após inserção
```

### Via Python:
```python
import asyncio
from scripts.ingest_episode import ingest_episode

stats = asyncio.run(ingest_episode(
    transcript_path="data/transcripts/ep001.txt",
    episode_id="ep001",
    title="The Fall of the Roman Republic",
    episode_number=1,
    episode_context="Roman politics, Caesar, civil war",
    run_communities=True,
))
```

## Pré-requisitos

- `OPENAI_API_KEY` no `.env`
- Neo4j rodando (`docker compose up -d`)
- `NEO4J_PASSWORD` no `.env`
- Arquivo de transcrição em texto plano

## Output (estatísticas)

```python
{
    "raw_chars": 45000,
    "clean_chars": 41000,
    "segments": 8,
    "chunks": 28,
    "entities_extracted": 156,
    "relationships_extracted": 89,
    "claims_extracted": 42,
    "entities_merged": 23,
    "unique_entities": 133,
    "upserted": {"entities": 133, "relationships": 89, "claims": 42},
    "embeddings": 203,
    "communities": {"communities_created": 12},  # se --communities
}
```

## Custo por episódio

| Etapa | Custo estimado |
|-------|----------------|
| Chunking (Level 1 segmentation) | ~$0.005 |
| Extraction (Passes 1-4) | ~$0.08 |
| Entity resolution (Stage 3-4) | ~$0.005 |
| Embeddings | ~$0.01 |
| Community summarization | ~$0.005 |
| **Total** | **~$0.10** |

## Idempotência

Rodar o script duas vezes no mesmo episódio:
- Extração: cache hit (zero chamadas API)
- Upsert: MERGE encontra nós existentes (zero duplicatas)
- Embeddings: sobrescreve com os mesmos valores
- Resultado: grafo idêntico, custo zero na segunda execução

## Fluxo de dados interno

```
transcript.txt
    → clean_transcript() → texto limpo
    → chunk_transcript() → list[Chunk]
    → process_episode() → {entities, relationships, claims, coreferences}
    → collect_entity_candidates() → list[EntityCandidate]
    → resolve_entities() → merge_map
    → upsert_episode_data() → dados no Neo4j
    → generate_embeddings() → vetores nos nós
    → detect_and_summarize_communities() → Community nodes
```
