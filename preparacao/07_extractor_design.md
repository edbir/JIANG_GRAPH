# Design do Extractor — Orquestrador Multi-Pass

## O que é o Extractor

O extractor é o **coração do pipeline de ingestion**. Ele é o módulo que de fato chama a OpenAI API, passando os chunks de texto junto com os prompts e schemas que já definimos, e coleta os resultados estruturados.

Pense nele como um maestro: ele não toca nenhum instrumento (não faz extração ele mesmo), mas coordena quem toca o quê e em que ordem.

## Onde ele se encaixa

```
text_cleaner.py → chunker.py → [extractor.py] → resolver.py → graph/
                                      ↑
                              USA: prompts.py (o que perguntar)
                              USA: schemas.py (formato da resposta)
                              USA: config.py (qual modelo, thresholds)
```

## O que ele faz, passo a passo

### Para cada chunk de um episódio:

1. **Pass 1 — Entity Extraction** (GPT-4o-mini)
   - Input: chunk text + contexto do episódio
   - Output: lista de entidades (Person, Organization, Topic, Event)
   - Custo: barato (modelo mini, structured output)

2. **Pass 2 — Relationship Extraction** (GPT-4o-mini)
   - Input: mesmo chunk + entidades do Pass 1
   - Output: lista de relações entre as entidades
   - Por que recebe entidades: para focar em relações entre entidades CONHECIDAS

3. **Pass 3 — Claim Extraction** (GPT-4o)
   - Input: mesmo chunk + entidades do Pass 1
   - Output: lista de claims, interpretações, previsões
   - Por que GPT-4o: requer nuance para distinguir fato de opinião de previsão

### Após TODOS os chunks do episódio:

4. **Pass 4 — Coreference Resolution** (GPT-4o)
   - Input: janela deslizante de 3-5 chunks + todas as entidades desses chunks
   - Output: mapeamento de referências ambíguas → entidades canônicas
   - Por que após todos os chunks: precisa da lista completa de entidades do episódio
   - Por que janela: pronomes referem entidades de chunks anteriores

### Após TODOS os episódios (não no extractor):

5. **Pass 5 — Entity Resolution** (GPT-4o-mini, só ~5% dos casos)
   - Chamado externamente pelo resolver.py
   - O extractor apenas expõe a função `resolve_entity_pair()`

## Decisões de Design

### Por que sequencial (não paralelo)?

Processamos chunks sequencialmente (um após o outro) em vez de em paralelo. Motivo: **prompt caching**.

A OpenAI cacheia prefixos de prompt repetidos (>1024 tokens) e dá 50% de desconto. Como nosso system prompt é idêntico para todos os chunks de um episódio, processar sequencialmente mantém o cache "quente" — cada chamada subsequente paga metade do preço no system prompt.

Se processássemos em paralelo, as chamadas chegariam em ordem aleatória e o cache seria menos efetivo.

### Por que caching de respostas?

Toda chamada à API é cacheada localmente por `hash(system_prompt + user_prompt + model)`. Isso significa:

- **Reprocessar é grátis:** Se rodarmos o pipeline de novo no mesmo episódio, zero chamadas à API.
- **Debugging fácil:** Podemos inspecionar o que o LLM retornou para cada chunk em `data/cache/`.
- **Idempotência:** Mesmo input → mesmo output, sempre.
- **Proteção contra falhas:** Se o pipeline crashar no chunk 15 de 30, ao reiniciar ele pula os 14 já cacheados.

### Por que uma única função `_call_llm()`?

Todos os 5 passes usam a mesma função para chamar a API. Isso centraliza:
- A lógica de caching
- O formato de chamada (structured outputs)
- O tratamento de erros (futuro)
- O logging (futuro)

Se precisarmos mudar algo na forma como chamamos a API (ex: adicionar retries, rate limiting), mudamos em UM lugar.

### Por que Pass 4 usa janela deslizante?

O coreference resolution precisa de contexto além do chunk individual. Exemplo:

```
Chunk 5: "Alexander the Great conquered the Persian Empire in 331 BC."
Chunk 6: "He then marched his army eastward toward India."
Chunk 7: "The Macedonian king's ambition seemed limitless."
```

Se processarmos chunk 6 sozinho, não sabemos quem é "He". Precisamos ver chunk 5 junto.

A janela deslizante (default: 4 chunks) garante que temos contexto suficiente. As janelas se sobrepõem por 1 chunk para não perder referências na fronteira.

### Por que Pass 5 não roda aqui?

Entity resolution compara entidades ENTRE episódios. Não faz sentido rodar dentro de `process_episode()` porque:
- Precisamos de TODAS as entidades de TODOS os episódios primeiro
- A cascata (exact → fuzzy → embedding → LLM) é um processo separado
- Só ~5% dos pares chegam ao LLM — o resto é resolvido sem API

## Estrutura do Output

`process_episode()` retorna um dicionário com tudo extraído:

```python
{
    "entities": {
        "ep001-seg00-chunk00": EntityExtractionResult(...),
        "ep001-seg00-chunk01": EntityExtractionResult(...),
        ...
    },
    "relationships": {
        "ep001-seg00-chunk00": RelationshipExtractionResult(...),
        ...
    },
    "claims": {
        "ep001-seg00-chunk00": ClaimExtractionResult(...),
        ...
    },
    "coreferences": [
        CoreferenceResult(...),  # janela 1
        CoreferenceResult(...),  # janela 2
        ...
    ],
}
```

Esse dicionário é o que alimenta o próximo passo: entity resolution e depois upsert no Neo4j.

## Estimativa de Custo por Episódio

Para um episódio de ~1 hora (~30 chunks de 1500 tokens):

| Pass | Modelo | Chamadas | Custo estimado |
|------|--------|----------|----------------|
| Pass 1 (entities) | gpt-4o-mini | 30 | ~$0.01 |
| Pass 2 (relationships) | gpt-4o-mini | 30 | ~$0.01 |
| Pass 3 (claims) | gpt-4o | 30 | ~$0.04 |
| Pass 4 (coreference) | gpt-4o | ~8 janelas | ~$0.02 |
| **Total por episódio** | | **~98 chamadas** | **~$0.08** |

Para 5 episódios (MVP): ~$0.40
Para 134 episódios (full): ~$10.70

Esses valores são estimativas conservadoras. Com prompt caching (50% desconto no system prompt), o custo real será menor.

## Notas para o TCC

### O que documentar no artigo:
1. **A arquitetura multi-pass** — por que é superior a single-pass (com números da pesquisa)
2. **O papel do structured output** — garantia de 100% conformidade JSON
3. **A estratégia de caching** — idempotência e economia
4. **O tiered model selection** — modelo barato para volume, modelo forte para nuance
5. **O processamento sequencial** — trade-off entre velocidade e custo (prompt caching)
6. **A janela deslizante do Pass 4** — por que coreference precisa de contexto multi-chunk
