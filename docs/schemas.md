# Schemas de Extração — Documentação Técnica

## O que é

`src/extraction/schemas.py` define os modelos Pydantic que especificam EXATAMENTE o que cada pass de extração produz. São passados à OpenAI via `response_format` para garantir 100% de conformidade JSON.

## Por que Pydantic + Structured Outputs

- **Sem Structured Outputs:** O LLM retorna JSON "best effort" — pode ter campos faltando, tipos errados, ou formato inválido (~60% de falha com GPT-4-0613).
- **Com Structured Outputs:** A OpenAI usa constrained decoding para GARANTIR que o output segue o schema. 100% de conformidade.

O schema Pydantic é convertido em JSON Schema e passado na chamada da API. O LLM é fisicamente incapaz de retornar algo fora do formato.

## Os 5 schemas (um por pass)

### Pass 1: `EntityExtractionResult`
```
entities: list[Entity]
    - name: str (canônico)
    - entity_type: enum (person/organization/topic/event)
    - description: str
    - aliases: list[str]
    - historical_role: str | None (só person)
    - org_type: str | None (só organization)
    - topic_category: str | None (só topic)
    - event_type: str | None (só event)
```

### Pass 2: `RelationshipExtractionResult`
```
relationships: list[Relationship]
    - source_entity: str
    - target_entity: str
    - relationship_type: enum (ROLE_IN/PARTICIPATED_IN/LED_TO/CAUSED/CONTROLLED/LOCATED_IN)
    - description: str
    - confidence: float (0-1)
```

### Pass 3: `ClaimExtractionResult`
```
claims: list[Claim]
    - text: str (standalone statement)
    - claim_type: enum (factual_assertion/historical_interpretation/geopolitical_prediction/causal_analysis)
    - stance: enum (critical/favorable/neutral/speculative)
    - entities_involved: list[str]
    - time_reference: str | None
    - confidence: float (0-1)
    - target_date: str | None (só predictions)
    - target_description: str | None (só predictions)
```

### Pass 4: `CoreferenceResult`
```
coreferences: list[CoreferenceEntry]
    - reference_text: str ("he", "the empire")
    - resolved_entity: str ("Alexander the Great")
    - chunk_index: int
```

### Pass 5: `EntityResolutionBatch`
```
judgments: list[EntityResolutionJudgment]
    - entity_a: str
    - entity_b: str
    - should_merge: bool
    - canonical_name: str
    - reasoning: str
```

## Enums (valores restritos)

| Enum | Valores | Usado em |
|------|---------|----------|
| EntityType | person, organization, topic, event | Pass 1 |
| RelationshipType | ROLE_IN, PARTICIPATED_IN, LED_TO, CAUSED, CONTROLLED, LOCATED_IN | Pass 2 |
| ClaimType | factual_assertion, historical_interpretation, geopolitical_prediction, causal_analysis | Pass 3 |
| Stance | critical, favorable, neutral, speculative | Pass 3 |

## Decisões de design

- **Campos condicionais:** `historical_role` só existe para persons, `org_type` só para organizations, etc. Evita campos vazios.
- **Stance vs. Sentiment:** "Stance" é mais preciso para análise histórica que "positive/negative".
- **Jiang implícito:** Não há campo `speaker` — Jiang é a fonte de todas as claims.
- **Predictions como subtipo:** Não é um schema separado — é um Claim com `claim_type=geopolitical_prediction` e campos extras opcionais.
- **Confidence threshold:** Abaixo de 0.7 não é ingerido no grafo (configurável em config.py).
