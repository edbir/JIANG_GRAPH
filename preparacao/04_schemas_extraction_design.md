# Design dos Schemas de Extração

## Contexto

O pipeline de extração usa 5 passes sequenciais, cada um com um propósito específico e um modelo de dados (schema) que define exatamente o que é produzido. Usamos Pydantic para definir esses schemas, o que nos dá:

1. **Validação automática** — se o LLM retornar algo fora do schema, sabemos imediatamente
2. **Structured Outputs** — passamos o schema para a OpenAI API via `response_format`, garantindo 100% de conformidade JSON
3. **Documentação viva** — os modelos Pydantic servem como documentação do que cada pass produz

---

## Pass 1 — Entity Extraction

**Modelo:** GPT-4o-mini (barato, alto volume, structured output)  
**Input:** Um chunk de texto (Level 2, ~1500 tokens) + contexto do episódio  
**Output:** Lista de entidades encontradas no chunk

### Campos por entidade:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `name` | str | Nome canônico universalmente reconhecido (ex: "Alexander the Great", "Roman Empire") |
| `entity_type` | enum | Um de: `person`, `organization`, `topic`, `event` |
| `description` | str | Descrição breve extraída do contexto do chunk |
| `aliases` | list[str] | Nomes alternativos mencionados (ex: ["Alexander", "Alexander III of Macedon"]) |
| `historical_role` | str \| None | Só para person. Ex: ruler, military_leader, philosopher, diplomat, civilian |
| `org_type` | str \| None | Só para organization. Ex: empire, nation-state, dynasty, alliance, religion, institution |
| `topic_category` | str \| None | Só para topic. Ex: geography, ideology, economic_system, governing_system, military_strategy, religious_practice |
| `event_type` | str \| None | Só para event. Ex: war, battle, treaty, revolution, crisis, period |

### Decisões de design:
- **Nome canônico:** Usamos nomes universalmente reconhecidos para facilitar entity resolution posterior. "Alexander the Great" e não "Alexander" ou "o macedônio".
- **Aliases:** Capturamos todas as formas como a entidade é mencionada no texto. Isso alimenta o entity resolution no Pass 5.
- **Campos condicionais:** `historical_role`, `org_type`, `topic_category`, `event_type` são mutuamente exclusivos — cada um só se aplica ao seu `entity_type`. Isso evita campos vazios desnecessários.

---

## Pass 2 — Relationship Extraction

**Modelo:** GPT-4o-mini (mesmo chunk, entidades como contexto)  
**Input:** O mesmo chunk + lista de entidades extraídas no Pass 1  
**Output:** Lista de relações entre entidades

### Campos por relação:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `source_entity` | str | Nome da entidade de origem (deve corresponder a um `name` do Pass 1) |
| `target_entity` | str | Nome da entidade de destino (deve corresponder a um `name` do Pass 1) |
| `relationship_type` | enum | Um dos tipos definidos no schema do grafo (ver abaixo) |
| `description` | str | Descrição da relação em linguagem natural |
| `confidence` | float | 0.0 a 1.0 — confiança do LLM na extração |

### Tipos de relação permitidos:
- `ROLE_IN` — Person → Organization (liderança, afiliação)
- `PARTICIPATED_IN` — Person/Organization → Event (participação em guerras, tratados)
- `LED_TO` — Event → Event (causalidade histórica)
- `CAUSED` — Event → Event (causalidade direta)
- `CONTROLLED` — Organization → Topic (quando Topic é geografia/território)
- `LOCATED_IN` — Organization → Topic (localização geográfica)
- `MENTIONS` — implícito pela co-ocorrência no chunk, não precisa ser extraído explicitamente

### Decisões de design:
- **Entidades como contexto:** Injetar as entidades do Pass 1 no prompt do Pass 2 faz o modelo focar em relações entre entidades conhecidas, em vez de re-extrair entidades.
- **Tipos restritos:** Limitamos os tipos de relação ao schema do grafo. Isso é schema-constrained extraction — o LLM não inventa tipos novos.
- **Confidence:** Permite filtrar relações de baixa confiança antes de inserir no grafo.

---

## Pass 3 — Claim/Opinion Extraction

**Modelo:** GPT-4o (necessário para nuance de stance e interpretação)  
**Input:** O mesmo chunk + entidades do Pass 1  
**Output:** Lista de claims feitas pelo Prof. Jiang

### Campos por claim:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `text` | str | A claim como uma afirmação standalone clara |
| `claim_type` | enum | Um de: `factual_assertion`, `historical_interpretation`, `geopolitical_prediction`, `causal_analysis` |
| `stance` | enum | Um de: `critical`, `favorable`, `neutral`, `speculative` |
| `entities_involved` | list[str] | Nomes de entidades referenciadas pela claim |
| `time_reference` | str \| None | Período histórico ou data futura referenciada |
| `confidence` | float | 0.0 a 1.0 — confiança na extração |

### Campos adicionais para predictions (quando `claim_type == geopolitical_prediction`):

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `target_date` | str \| None | Data alvo da previsão, se mencionada |
| `target_description` | str \| None | O que se espera que aconteça |

### Decisões de design:
- **Claims como nós (reificação):** Claims não são propriedades de arestas — são nós first-class no grafo. Isso permite anexar metadados, rastrear evolução, e modelar relações claim-to-claim (SUPPORTS, CONTRADICTS).
- **Stance vs. Sentiment:** Usamos "stance" em vez de "sentiment" porque é mais preciso para análise histórica. "Critical" e "favorable" capturam a posição do Jiang sobre um tema melhor que "positive/negative".
- **Jiang implícito:** Não temos campo `speaker` porque Jiang é a fonte implícita de todas as claims. Isso simplifica o schema sem perder informação.
- **Predictions como subtipo:** Em vez de um schema separado, predictions são claims com `claim_type == geopolitical_prediction` e campos adicionais opcionais. Isso mantém o schema unificado.

---

## Pass 4 — Coreference Resolution

**Modelo:** GPT-4o (requer raciocínio contextual profundo)  
**Input:** Entidades do Pass 1 numa janela deslizante de 3-5 chunks  
**Output:** Cadeias de correferência mapeando referências ambíguas a entidades canônicas

### Campos por correferência:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `reference_text` | str | O texto ambíguo no chunk (ex: "him", "the empire", "that religion") |
| `resolved_entity` | str | Nome canônico da entidade referida |
| `chunk_index` | int | Em qual chunk (da janela) a referência aparece |

### Decisões de design:
- **Janela deslizante:** Pronomes frequentemente referem entidades mencionadas em chunks anteriores. Uma janela de 3-5 chunks captura a maioria das correferências cross-chunk.
- **Crítico para inglês:** Em inglês conversacional, pronomes e referências indiretas são muito frequentes. "He then conquered the city" — quem é "he"? Qual "city"? Sem coreference resolution, essas informações se perdem.
- **Alimenta entity resolution:** As correferências resolvidas ajudam o Pass 5 a entender que "the Macedonian" e "Alexander the Great" são a mesma entidade.

---

## Pass 5 — Entity Resolution (LLM judgment stage)

**Modelo:** GPT-4o-mini (só para os ~5% de casos ambíguos)  
**Input:** Par de entidades com seus contextos (nome, descrição, aliases, relações)  
**Output:** Decisão de merge

### Campos:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `entity_a` | str | Nome da primeira entidade |
| `entity_b` | str | Nome da segunda entidade |
| `should_merge` | bool | Se as duas entidades são a mesma coisa |
| `canonical_name` | str | Qual nome usar como canônico se merge = true |
| `reasoning` | str | Justificativa breve da decisão |

### Decisões de design:
- **Cascata antes do LLM:** Antes de chegar aqui, 95% dos duplicados já foram resolvidos por: (1) exact match normalizado, (2) fuzzy string matching (RapidFuzz > 85), (3) embedding similarity (cosine > 0.85). Só os casos genuinamente ambíguos chegam ao LLM.
- **Reasoning:** Forçar o LLM a justificar ajuda na qualidade da decisão e permite auditoria humana posterior.
- **Graph-aware:** Após a resolução inicial, um segundo passo verifica entidades que compartilham >3 relações com os mesmos alvos — provavelmente são duplicatas que os métodos anteriores não pegaram.

---

## Visão Geral do Fluxo

```
Chunk de texto
    │
    ▼
┌─────────────────────┐
│ Pass 1: Entidades    │ → Lista de Entity
└──────────┬──────────┘
           │ (entidades como contexto)
           ▼
┌─────────────────────┐
│ Pass 2: Relações     │ → Lista de Relationship
└──────────┬──────────┘
           │ (entidades como contexto)
           ▼
┌─────────────────────┐
│ Pass 3: Claims       │ → Lista de Claim
└──────────┬──────────┘
           │ (janela de 3-5 chunks)
           ▼
┌─────────────────────┐
│ Pass 4: Coreference  │ → Lista de CoreferenceChain
└──────────┬──────────┘
           │ (após todos os chunks do episódio)
           ▼
┌─────────────────────┐
│ Pass 5: Resolution   │ → Decisões de Merge
└─────────────────────┘
```

---

## Notas para o TCC

Esta seção do design alimenta diretamente a **Seção 4 (Desenvolvimento)** do artigo, onde explicaremos:
- Por que multi-pass é superior a single-pass (ECR: +16% F1, KGGen: +18% sobre GraphRAG)
- Por que structured outputs garantem conformidade (100% vs <40% sem constrained decoding)
- Por que claims são nós e não propriedades de arestas (reificação, AIF, Kialo)
- Por que a cascata de entity resolution é necessária (CORE-KG: 28.32% mais duplicação sem coreference)
