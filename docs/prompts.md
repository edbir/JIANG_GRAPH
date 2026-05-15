# Prompts de Extração — Documentação Técnica

## O que é

`src/extraction/prompts.py` contém todos os prompts usados nos 5 passes de extração. Cada pass tem um SYSTEM prompt (instruções estáticas) e um USER template (conteúdo variável).

## Arquitetura dos prompts

```
SYSTEM (estático, cacheável pela OpenAI — 50% desconto):
    - Quem o LLM é (role/expertise)
    - O domínio (Predictive History, Prof. Jiang)
    - Os tipos permitidos (schema-constrained)
    - As regras de extração

USER (variável, muda a cada chunk):
    - O texto do chunk
    - Contexto adicional (entidades já extraídas, info do episódio)
```

## Por que o formato de retorno NÃO está no prompt

O formato é definido via `response_format` na chamada da API (em extractor.py), não no prompt. Isso porque:
1. Structured Outputs garante 100% conformidade sem instruções no prompt
2. O prompt fica mais limpo e focado na tarefa semântica
3. Não desperdiçamos tokens com "return a JSON with fields..."

## Os 5 prompts

### Pass 1: Entity Extraction
- **System:** Expert historian, lista entity types e rules
- **User:** Episode context + chunk text
- **Modelo:** gpt-4o-mini

### Pass 2: Relationship Extraction
- **System:** Expert historian, lista relationship types com exemplos
- **User:** Entidades do Pass 1 + chunk text
- **Modelo:** gpt-4o-mini

### Pass 3: Claim/Opinion Extraction
- **System:** Expert in argumentation, distingue 4 tipos de claim, explica stance
- **User:** Entidades do Pass 1 + chunk text
- **Modelo:** gpt-4o (nuance necessária)

### Pass 4: Coreference Resolution
- **System:** Linguistic analyst, tipos de referência ambígua
- **User:** Entidades de todos os chunks na janela + chunks numerados
- **Modelo:** gpt-4o (raciocínio contextual)

### Pass 5: Entity Resolution
- **System:** Knowledge graph curator, sinais de merge vs. keep separate
- **User:** Duas entidades com todos os seus atributos
- **Modelo:** gpt-4o-mini (decisão binária)

## Princípios de design

1. **Schema-constrained:** Lista explicitamente os tipos permitidos. O LLM não inventa categorias.
2. **Domain-specific:** Menciona Prof. Jiang, história, geopolítica. Não é genérico.
3. **Context-enriched:** Cada pass recebe output dos passes anteriores.
4. **Temperature 0.0:** Determinismo e reprodutibilidade.
5. **Prompt caching:** System prompt idêntico para todos os chunks → 50% desconto.

## Prompt de segmentação (no chunker)

Além dos 5 prompts de extração, existe um prompt de segmentação em `chunker.py`:
- **System:** Analista de estrutura de discurso, identifica mudanças de tópico
- **User:** Sentenças numeradas de uma janela
- **Modelo:** gpt-4o-mini
- **Output:** Lista de índices onde tópicos mudam
