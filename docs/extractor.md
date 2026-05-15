# Extractor — Documentação Técnica

## O que é

`src/extraction/extractor.py` é o orquestrador que chama a OpenAI API com os prompts e schemas corretos para cada pass, gerencia caching, e coordena o fluxo entre passes.

## Posição no pipeline

```
chunker.py → list[Chunk] → [extractor.py] → extracted knowledge → resolver.py → graph/
                                  ↑
                          Usa: prompts.py + schemas.py + config.py
```

## O que faz

Para cada chunk de um episódio, roda sequencialmente:
1. **Pass 1:** Entity Extraction (gpt-4o-mini)
2. **Pass 2:** Relationship Extraction (gpt-4o-mini, recebe entidades do Pass 1)
3. **Pass 3:** Claim Extraction (gpt-4o, recebe entidades do Pass 1)

Após todos os chunks:
4. **Pass 4:** Coreference Resolution (gpt-4o, janela deslizante de 4 chunks)

Expõe para uso externo:
5. **Pass 5:** `resolve_entity_pair()` — chamado pelo resolver.py quando necessário

## Funções principais

| Função | Pass | Modelo | Input | Output |
|--------|------|--------|-------|--------|
| `extract_entities()` | 1 | gpt-4o-mini | chunk + episode context | EntityExtractionResult |
| `extract_relationships()` | 2 | gpt-4o-mini | chunk + entities | RelationshipExtractionResult |
| `extract_claims()` | 3 | gpt-4o | chunk + entities | ClaimExtractionResult |
| `resolve_coreferences()` | 4 | gpt-4o | window of chunks + entities | CoreferenceResult |
| `resolve_entity_pair()` | 5 | gpt-4o-mini | two entity candidates | EntityResolutionBatch |
| `process_episode()` | ALL | mixed | list[Chunk] | dict with all results |

## Caching

Toda chamada LLM é cacheada em `data/cache/` por `sha256(system + user + model)`.

Benefícios:
- Reprocessar = $0 (cache hit)
- Debugging: inspecionar respostas em `data/cache/*.json`
- Idempotência: mesmo input → mesmo output
- Crash recovery: reiniciar pula chunks já processados

## Por que sequencial (não paralelo)

Processamento sequencial mantém o prompt cache da OpenAI "quente":
- System prompt é idêntico para todos os chunks
- OpenAI cacheia prefixos >1024 tokens repetidos
- Desconto de 50% nos tokens do system prompt
- Paralelo quebraria esse cache

## Output de `process_episode()`

```python
{
    "entities": {chunk_id: EntityExtractionResult, ...},
    "relationships": {chunk_id: RelationshipExtractionResult, ...},
    "claims": {chunk_id: ClaimExtractionResult, ...},
    "coreferences": [CoreferenceResult, ...],
}
```

## Custo estimado

| Pass | Modelo | Chamadas/episódio | Custo |
|------|--------|-------------------|-------|
| 1 (entities) | gpt-4o-mini | ~30 | ~$0.01 |
| 2 (relationships) | gpt-4o-mini | ~30 | ~$0.01 |
| 3 (claims) | gpt-4o | ~30 | ~$0.04 |
| 4 (coreference) | gpt-4o | ~8 | ~$0.02 |
| **Total/episódio** | | **~98** | **~$0.08** |

MVP (5 ep): ~$0.40 | Full (134 ep): ~$10.70
