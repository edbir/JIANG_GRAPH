# Preparação — Seção 5: Avaliação

## Objetivo da Seção
Demonstrar que o sistema funciona e quantificar sua qualidade. A banca quer ver métricas, comparações, e análise honesta de limitações. Não precisa ser perfeito — precisa ser medido e discutido.

---

## Estrutura Planejada

### 5.1 Metodologia de Avaliação
**O que cobrir:**
- Como avaliamos o sistema (quais métricas, quais benchmarks)
- Divisão em: qualidade de extração, qualidade de retrieval, utilidade end-to-end
- O corpus de teste: quais episódios, quantos, por quê esses
- Limitações da avaliação (sem ground truth anotado manualmente para todo o corpus)

**Sugestões de estudo:**
- Pesquisar "evaluation metrics for knowledge graph extraction" (precision, recall, F1)
- Pesquisar "RAG evaluation metrics" (faithfulness, relevance, completeness)
- O framework MINE do KGGen (NeurIPS 2025): MINE-1 (knowledge retention) + MINE-2 (RAG effectiveness)
- RAGAS framework para avaliação de RAG

### 5.2 Avaliação da Extração
**O que cobrir:**
- Sampling-based validation: selecionar 5-10 episódios e verificar manualmente
- Métricas por dimensão:
  - Entity extraction recall (entidades nomeadas estão sendo encontradas?)
  - Relationship correctness (relações extraídas refletem o que foi dito?)
  - Claim attribution accuracy (claims são fiéis ao que Jiang disse?)
  - Stance labeling accuracy (critical/favorable/neutral/speculative corretos?)
- Target: >85% accuracy across all dimensions
- Análise de erros: que tipos de erros o sistema comete? Por quê?

**Base na implementação:**
- Ver `scripts/validate_graph.py` (queries de validação estrutural)
- Ver `tests/test_queries.md` (queries para testar retrieval)
- Os confidence thresholds (0.7) filtram extrações de baixa qualidade

**Sugestões de estudo:**
- Pesquisar "human evaluation of information extraction"
- Entender inter-annotator agreement (se possível, ter outra pessoa validar)
- O paper SAC-KG mostra como avaliar precision em KG extraction

### 5.3 Avaliação do Retrieval
**O que cobrir:**
- Testar com as queries de `tests/test_queries.md` (100 queries, 4 tipos)
- Métricas:
  - Relevance: o contexto retornado é relevante para a pergunta?
  - Completeness: informação importante está faltando?
  - Precision: há informação irrelevante no contexto?
- Comparação: nosso sistema vs. RAG tradicional (mesmas queries, mesmo corpus)
  - RAG baseline: chunks com embedding search, sem grafo
  - Nosso sistema: vector + graph fan-out + claims + communities
- Análise por tipo de query: onde o grafo ajuda mais? Onde não faz diferença?

**Sugestões de estudo:**
- Pesquisar "RAG evaluation" e "retrieval evaluation metrics"
- O paper GraphRAG-Bench (ICLR 2026) mostra quando grafos ajudam e quando não
- Pesquisar "NDCG" e "MRR" como métricas de ranking

### 5.4 Avaliação End-to-End (Utilidade)
**O que cobrir:**
- Testar com usuários reais (mesmo que seja só você + orientador)
- Perguntas qualitativas:
  - O sistema responde perguntas que RAG tradicional não consegue?
  - As respostas citam fontes (episódios, momentos)?
  - O sistema admite quando não sabe?
  - As comparações entre entidades são úteis?
  - O timeline de previsões funciona?
- Exemplos concretos: mostrar queries e respostas lado a lado (nosso sistema vs. baseline)

**Sugestões de estudo:**
- Pesquisar "user study evaluation" em sistemas de IR/NLP
- Não precisa ser formal (questionário com N participantes) — pode ser análise qualitativa com exemplos

### 5.5 Análise de Custos
**O que cobrir:**
- Custo real de extração (quanto gastou nos 5 episódios do MVP)
- Custo projetado para 134 episódios
- Comparação com custo documentado do Microsoft GraphRAG ($33K)
- Breakdown por componente (extração, embeddings, communities, retrieval)

**Base na implementação:**
- Ver `preparacao/03_infraestrutura_e_dependencias.md` (estimativas)
- Ver `docs/extractor.md` (custo por episódio)
- Registrar custos reais durante o MVP

**Sugestões de estudo:**
- Documentar TODOS os custos durante o desenvolvimento (OpenAI dashboard)
- Comparar com custos reportados na literatura

### 5.6 Limitações e Ameaças à Validade
**O que cobrir:**
- Limitações do sistema:
  - Dependência de um único LLM provider (OpenAI)
  - Qualidade da transcrição afeta tudo downstream
  - Single speaker (Jiang) — não testado com múltiplos speakers
  - Corpus em inglês — não testado com outros idiomas
  - Sem ground truth anotado por especialistas em história
- Ameaças à validade:
  - Avaliação feita pelo próprio desenvolvedor (bias)
  - Corpus limitado (5 episódios no MVP)
  - Prompts podem estar over-fitted para o estilo do Jiang

**Sugestões de estudo:**
- Pesquisar "threats to validity" em artigos de engenharia de software
- Ser HONESTO sobre limitações — a banca valoriza autocrítica mais que perfeição

---

## Dica para a escrita
A avaliação deve ser HONESTA. Mostrar onde o sistema funciona bem E onde falha. Analisar os erros é tão valioso quanto reportar os acertos. A banca quer ver pensamento crítico, não marketing.
