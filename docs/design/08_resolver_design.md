# Design do Entity Resolver — Cascata de Deduplicação

## O Problema

Após rodar os Passes 1-4 em todos os chunks de todos os episódios, temos uma lista grande de entidades extraídas. O problema: a mesma entidade real aparece com nomes diferentes.

Exemplos reais que vão acontecer no corpus do Jiang:
- "Alexander the Great" / "Alexander" / "Alexander III" / "the Macedonian king"
- "Roman Empire" / "Rome" / "the Romans" / "the Empire"
- "United States" / "the US" / "USA" / "America"
- "Philip II" / "Philip of Macedon" / "Philip" / "Alexander's father"

Se não resolvermos isso, o grafo terá 4 nós separados para Alexander the Great, cada um com suas relações e claims. Isso destrói a utilidade do grafo — queries como "tudo que Jiang disse sobre Alexander" retornariam apenas uma fração dos resultados.

## A Solução: Cascata de Resolução (Cheap → Expensive)

A ideia central: **não usar LLM para tudo**. A maioria dos duplicados é óbvia e pode ser resolvida com métodos baratos. Só os casos genuinamente ambíguos vão para o LLM.

A cascata tem 4 estágios, cada um mais caro que o anterior:

### Estágio 1 — Exact Match (~40% dos duplicados)

**O que faz:** Normaliza nomes (lowercase, remove espaços extras, remove acentos) e compara.

**Exemplos que resolve:**
- "United States" = "united states" = "UNITED STATES"
- "Roman Empire" = "Roman  Empire" (espaço duplo)
- "São Paulo" = "Sao Paulo" (acento)

**Custo:** Zero. É string comparison pura.

### Estágio 2 — Fuzzy String Matching (~35% dos duplicados)

**O que faz:** Usa RapidFuzz (token_sort_ratio) para comparar nomes de entidades do MESMO TIPO. Score > 85 = provável duplicata.

**Exemplos que resolve:**
- "Philip the Second" ≈ "Philip II" (score ~88)
- "Peloponnesian War" ≈ "The Peloponnesian War" (score ~92)
- "Byzantine Empire" ≈ "Byzantium Empire" (score ~90)

**Custo:** Negligível. RapidFuzz é implementado em C, processa milhares de pares por segundo.

**Restrição importante:** Só compara entidades do MESMO tipo. Uma person nunca é comparada com um event, mesmo que os nomes sejam similares.

### Estágio 3 — Embedding Similarity (~20% dos duplicados)

**O que faz:** Gera embeddings de "nome + descrição" de cada entidade e compara por cosine similarity. Score > 0.85 = provável duplicata.

**Exemplos que resolve:**
- "USA" ≈ "United States" (nomes diferentes, mas descrições similares)
- "WWI" ≈ "World War I" (abreviação vs. nome completo)
- "The Great War" ≈ "World War I" (nome histórico vs. moderno)

**Custo:** Uma chamada de embedding por entidade (barato: ~$0.02/1M tokens). A comparação em si é multiplicação de vetores.

**⚠️ Requer API:** Este estágio precisa da OpenAI API para gerar embeddings.

### Estágio 4 — LLM Judgment (~5% dos duplicados)

**O que faz:** Para pares genuinamente ambíguos que os métodos anteriores não resolveram, pergunta ao GPT-4o-mini: "essas duas entidades são a mesma coisa?"

**Exemplos que resolve:**
- "Philip II" (person, "King of Macedon") vs. "Philip II" (person, "King of Spain") → NÃO merge (pessoas diferentes!)
- "The Republic" (org, "Roman political system") vs. "Roman Republic" (org, "state in ancient Italy") → merge

**Custo:** Uma chamada LLM por par ambíguo. Como só ~5% chega aqui, são poucas chamadas.

## Princípio Conservador: Na Dúvida, NÃO Merge

Em knowledge graphs, um falso positivo (merge incorreto) é MUITO pior que um falso negativo (manter duplicatas):

- **Falso positivo:** Mergeamos "Philip II of Macedon" com "Philip II of Spain". Agora TODAS as claims e relações de ambos estão misturadas. O grafo está corrompido e é difícil desfazer.
- **Falso negativo:** Mantemos dois nós para "Alexander the Great" e "Alexander III". O grafo tem uma duplicata, mas nenhuma informação está errada. Podemos corrigir depois.

Por isso, todos os thresholds são conservadores (85 para fuzzy, 0.85 para embedding) e o LLM é instruído a não mergear na dúvida.

## Passo Extra: Graph-Aware Deduplication

Após a cascata inicial, rodamos um passo adicional que usa a ESTRUTURA do grafo:

Se duas entidades do mesmo tipo compartilham >3 relações com os mesmos alvos, provavelmente são a mesma entidade — mesmo que seus nomes não sejam similares o suficiente para os métodos anteriores.

Exemplo: Se "The Conqueror" e "Alexander the Great" ambos têm relações PARTICIPATED_IN com "Battle of Gaugamela", "Battle of Issus", e "Siege of Tyre", é quase certo que são a mesma pessoa.

## Output do Resolver

O resolver produz um **mapeamento de merge**:

```python
{
    "Alexander": "Alexander the Great",           # alias → canonical
    "Alexander III": "Alexander the Great",       # alias → canonical
    "the Macedonian king": "Alexander the Great", # alias → canonical
    "Roman Empire": "Roman Empire",               # já é canonical (sem merge)
    "Rome": "Roman Empire",                       # alias → canonical (quando se refere ao estado)
}
```

Este mapeamento é aplicado a TODAS as entidades, relações, e claims antes do upsert no Neo4j. Assim, o grafo final tem um único nó por entidade real.

## Notas para o TCC

### O que documentar no artigo:
1. **O problema de entity resolution em KGs** — por que é necessário
2. **A cascata cheap → expensive** — com percentuais de resolução por estágio
3. **O princípio conservador** — falso positivo vs. falso negativo em grafos
4. **Graph-aware deduplication** — usar estrutura do grafo como sinal adicional
5. **Referências:** CORE-KG (28.32% mais duplicação sem coreference), LINK-KG (45% redução com cascata)
