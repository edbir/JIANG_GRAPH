# Entity Resolver — Documentação Técnica

## O que é

`src/extraction/resolver.py` deduplica entidades que representam a mesma coisa do mundo real mas foram extraídas com nomes diferentes.

## Posição no pipeline

```
extractor.py → extracted entities → [resolver.py] → merge map → upsert.py
```

## O problema

Após extração, a mesma entidade aparece com nomes diferentes:
- "Alexander the Great" / "Alexander" / "Alexander III" / "the Macedonian king"
- "United States" / "the US" / "USA" / "America"

Sem resolução, o grafo teria múltiplos nós para a mesma entidade, fragmentando relações e claims.

## A cascata (4 estágios, cheap → expensive)

| Estágio | Método | % resolvido | Custo | Precisa API? |
|---------|--------|-------------|-------|-------------|
| 1 | Exact match (normalizado) | ~40% | Zero | Não |
| 2 | Fuzzy string (RapidFuzz) | ~35% | Zero | Não |
| 3 | Embedding similarity | ~20% | Barato | Sim |
| 4 | LLM judgment | ~5% | Moderado | Sim |

### Estágio 1 — Exact Match
Normaliza (lowercase, strip, remove acentos) e compara. Também verifica aliases.

### Estágio 2 — Fuzzy Match
RapidFuzz `token_sort_ratio` > 85 entre entidades do MESMO tipo.
Token sort lida com reordenamento: "Philip the Second" ≈ "the Second Philip".

### Estágio 3 — Embedding Similarity
Embeds "nome: descrição", compara por cosine > 0.85.
Pega casos onde nomes são diferentes mas semântica é igual.

### Estágio 4 — LLM Judgment
Só para ~5% genuinamente ambíguos. Usa `extractor.resolve_entity_pair()`.

## Princípio conservador

**Na dúvida, NÃO merge.** Falso positivo (merge errado) corrompe o grafo. Falso negativo (duplicata) é inofensivo e corrigível.

## Output

```python
merge_map: dict[str, str]
# {"Alexander": "Alexander the Great", "USA": "United States", ...}
```

Resolve cadeias transitivas: se A→B e B→C, então A→C.

## Como usar

```python
from src.extraction.resolver import resolve_entities, EntityCandidate

candidates = [EntityCandidate(name="Alexander", ...), ...]
merge_map, decisions = await resolve_entities(candidates, client=openai_client)

# Aplicar merge_map a todas as entidades/relações/claims antes do upsert
```

## Estruturas de dados

```python
@dataclass
class EntityCandidate:
    name: str
    entity_type: str
    description: str
    aliases: list[str]
    source_chunks: list[str]

@dataclass
class MergeDecision:
    entity_a: str
    entity_b: str
    should_merge: bool
    canonical_name: str
    method: str          # "exact", "fuzzy", "embedding", "llm"
    score: float
    reasoning: str
```

## Restrição importante

**Nunca compara entidades de tipos diferentes.** Uma Person nunca é comparada com um Event, mesmo que nomes sejam similares. Isso evita merges absurdos.
