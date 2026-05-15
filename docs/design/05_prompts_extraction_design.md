# Design dos Prompts de Extração

## Contexto

Os prompts são a interface entre o texto bruto (transcrição) e os schemas Pydantic. Cada prompt instrui o LLM sobre:
1. **Quem ele é** (role/expertise)
2. **O que está analisando** (domínio, contexto do episódio)
3. **O que deve extrair** (tipos de entidade, regras específicas)
4. **Como deve formatar** (implícito via structured outputs, mas reforçado com regras)

A qualidade dos prompts determina diretamente a qualidade do knowledge graph. Prompts mal escritos produzem entidades genéricas, relações vagas, e claims superficiais.

---

## Princípios de Design dos Prompts

### 1. Schema-constrained
O prompt lista explicitamente os tipos permitidos (entity types, relationship types, claim types). O LLM não pode inventar categorias novas. Isso é o que a pesquisa chama de "ontology-guided prompting" e é responsável pela diferença entre 98.8% de precisão (ODKE+) e 40% de erro (extração aberta).

### 2. Domain-specific
Os prompts são escritos para o domínio de aulas de história e geopolítica. Não são prompts genéricos de "extract entities from text". Eles mencionam o Prof. Jiang, o tipo de conteúdo, e dão exemplos do domínio.

### 3. Context-enriched
Cada prompt recebe não apenas o chunk de texto, mas também contexto adicional:
- Pass 1: contexto do episódio (título, temas conhecidos)
- Pass 2: entidades já extraídas no Pass 1
- Pass 3: entidades do Pass 1 como referência
- Pass 4: entidades de múltiplos chunks (janela deslizante)

### 4. Few-shot quando necessário
Para passes mais complexos (claims, coreference), incluímos exemplos no prompt para calibrar o comportamento do LLM.

### 5. Temperature = 0.0
Todos os passes usam temperature 0 para máxima determinismo e reprodutibilidade. Não queremos criatividade na extração — queremos precisão.

---

## Pass 1 — Entity Extraction Prompt

### Objetivo
Identificar todas as entidades nomeadas no chunk, classificá-las, e fornecer descrições e aliases.

### Estrutura do prompt:
- **System message:** Define o role (expert historian/geopolitics analyst), o domínio (Predictive History), os tipos de entidade permitidos, e as regras de extração.
- **User message:** O chunk de texto + contexto do episódio.

### Regras-chave no prompt:
- Usar nomes canônicos universalmente reconhecidos
- Capturar TODOS os aliases e títulos mencionados
- Classificar persons por historical_role
- Tratar polities históricas como organizations (Roman Empire = organization, type=empire)
- Para topics, usar categorias planas (geography, ideology, etc.) sem taxonomia profunda
- Para events, incluir tipo e período

### Exemplo de extração esperada:
Texto: "Alexander conquered the Persian Empire after the Battle of Gaugamela in 331 BC"
→ Entities: Alexander the Great (person, military_leader), Persian Empire (organization, empire), Battle of Gaugamela (event, battle)

---

## Pass 2 — Relationship Extraction Prompt

### Objetivo
Identificar relações entre as entidades já extraídas no Pass 1, usando tipos restritos do schema.

### Estrutura do prompt:
- **System message:** Define o role, lista os tipos de relação permitidos com exemplos, e instrui a focar em relações explícitas no texto.
- **User message:** O chunk de texto + lista de entidades do Pass 1 (nome + tipo).

### Regras-chave no prompt:
- Só extrair relações entre entidades que foram identificadas no Pass 1
- Usar APENAS os tipos de relação do schema (ROLE_IN, PARTICIPATED_IN, LED_TO, CAUSED, CONTROLLED, LOCATED_IN)
- Cada relação deve ter evidência textual no chunk
- Atribuir confidence baseado na clareza da evidência
- Não inferir relações que não estão explícitas ou fortemente implícitas no texto

### Exemplo de extração esperada:
Texto: "Caesar, as dictator of Rome, led the campaign into Gaul"
Entidades disponíveis: [Julius Caesar (person), Roman Republic (org), Gallic Wars (event)]
→ Relationships:
  - Julius Caesar → ROLE_IN → Roman Republic (description: "served as dictator")
  - Julius Caesar → PARTICIPATED_IN → Gallic Wars (description: "led the military campaign")

---

## Pass 3 — Claim/Opinion Extraction Prompt

### Objetivo
Extrair as interpretações históricas, análises causais, previsões geopolíticas, e opiniões do Prof. Jiang. Este é o pass mais crítico — é o que diferencia nosso sistema de um simples extrator de entidades.

### Estrutura do prompt:
- **System message:** Define o role (expert in argumentation analysis), explica a diferença entre os tipos de claim, dá exemplos de cada tipo, e instrui sobre stance detection.
- **User message:** O chunk de texto + entidades do Pass 1 como referência.

### Regras-chave no prompt:
- Distinguir claramente entre:
  - **Factual assertions:** "Athens lost the Peloponnesian War" (fato histórico aceito)
  - **Historical interpretations:** "The fall of Rome was caused by economic inequality" (interpretação do Jiang)
  - **Geopolitical predictions:** "The US will face internal conflict within a decade" (previsão verificável)
  - **Causal analysis:** "The assassination of Archduke Ferdinand triggered WWI because..." (cadeia causal)
- Cada claim deve ser uma afirmação standalone (compreensível sem o contexto do chunk)
- Entities_involved deve mapear para entidades do Pass 1
- Stance reflete a posição do Jiang: ele é crítico? favorável? neutro? especulativo?
- Para predictions: capturar target_date e target_description quando mencionados

### Exemplo de extração esperada:
Texto: "I think the United States is following the same pattern as the late Roman Republic. The economic inequality, the political polarization — I predict we'll see some form of civil conflict within the next decade."
→ Claims:
  1. text: "The United States is following the same pattern as the late Roman Republic"
     type: historical_interpretation, stance: critical, entities: [United States, Roman Republic]
  2. text: "The US will experience civil conflict within the next decade"
     type: geopolitical_prediction, stance: speculative, entities: [United States]
     target_date: "within 10 years", target_description: "civil conflict in the United States"

---

## Pass 4 — Coreference Resolution Prompt

### Objetivo
Resolver referências ambíguas (pronomes, descrições indiretas) para entidades canônicas, usando uma janela de múltiplos chunks como contexto.

### Estrutura do prompt:
- **System message:** Define o role (linguistic analyst), explica o problema de coreference em transcrições faladas, lista os tipos comuns de referência ambígua.
- **User message:** A janela de 3-5 chunks numerados + lista de entidades conhecidas de todos os chunks na janela.

### Regras-chave no prompt:
- Resolver pronomes: "he", "him", "they", "it" → entidade específica
- Resolver descrições indiretas: "the empire", "that philosopher", "the city-state" → entidade específica
- Resolver referências contextuais: "the same thing happened there" → qual lugar?
- Só resolver quando há confiança razoável — não adivinhar
- Indicar o chunk_index onde cada referência aparece

### Exemplo de extração esperada:
Chunks na janela:
- [0]: "Alexander the Great conquered Persia after years of preparation."
- [1]: "He then marched his army eastward toward India."
- [2]: "The Macedonian king's ambition knew no bounds."

Entidades conhecidas: [Alexander the Great, Persian Empire, India]
→ Coreferences:
  - "He" (chunk 1) → Alexander the Great
  - "The Macedonian king" (chunk 2) → Alexander the Great
  - "his army" (chunk 1) → Alexander the Great (possessive reference)

---

## Pass 5 — Entity Resolution Prompt

### Objetivo
Decidir se dois candidatos a entidade são de fato a mesma entidade real, quando os métodos mais baratos (exact match, fuzzy, embedding) não conseguiram resolver.

### Estrutura do prompt:
- **System message:** Define o role (knowledge graph curator), explica o problema de entity resolution, instrui a considerar contexto e relações.
- **User message:** Os dois candidatos com seus contextos completos (nome, aliases, descrição, tipo, relações conhecidas).

### Regras-chave no prompt:
- Considerar aliases, descrições, e relações de ambas as entidades
- Entidades do mesmo tipo com descrições similares provavelmente são a mesma
- Entidades de tipos diferentes NUNCA devem ser merged
- Na dúvida, NÃO fazer merge (falso negativo é melhor que falso positivo em KGs)
- Sempre justificar a decisão

### Exemplo:
Entity A: name="Philip II", type=person, description="King of Macedon, father of Alexander"
Entity B: name="Philip of Macedon", type=person, description="Macedonian ruler who unified Greece"
→ should_merge: true, canonical_name: "Philip II of Macedon", reasoning: "Same historical figure — Philip II, king of Macedon and father of Alexander the Great. Different name forms for the same person."

---

## Notas sobre Prompt Engineering para o TCC

### O que documentar no artigo:
1. **A estrutura system/user message** — por que separamos instruções estáticas (system) de conteúdo variável (user)
2. **Schema-constrained vs. open extraction** — com os números da pesquisa (98.8% vs 40%)
3. **O papel do contexto** — como injetar entidades do Pass 1 no Pass 2 melhora a qualidade
4. **Temperature 0** — reprodutibilidade como requisito científico
5. **Iteração dos prompts** — documentar versões e melhorias ao longo do desenvolvimento

### Prompt caching (otimização de custo):
O system message é idêntico para todos os chunks de um episódio. A OpenAI oferece 50% de desconto em tokens de prefixo cacheado (>1024 tokens). Por isso, colocamos todas as instruções estáticas no system message e apenas o chunk variável no user message. Isso reduz custos significativamente quando processamos dezenas de chunks por episódio.
