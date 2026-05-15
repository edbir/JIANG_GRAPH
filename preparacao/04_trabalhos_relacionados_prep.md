# Preparação — Seção 3: Trabalhos Relacionados

## Objetivo da Seção
Posicionar nosso trabalho no contexto da literatura existente. Mostrar o que já foi feito, onde estão as lacunas, e como nosso sistema se diferencia. A banca quer ver que você conhece o campo e sabe onde seu trabalho se encaixa.

---

## Estrutura Planejada

### 3.1 Sistemas GraphRAG Existentes
**O que cobrir:**
- Microsoft GraphRAG (Edge et al., 2024): community detection + hierarchical summarization. Limitações: 4 entity types fixos, ~610K tokens por query global, custo de $33K para datasets grandes, sem suporte a dados conversacionais.
- LightRAG (Guo et al., HKU, 2024): extração aberta, ~100 tokens por query (6000× mais eficiente que GraphRAG), incremental via graph union. Limitações: zero schema constraints, 40% predicate errors, withdrawn de ICLR 2025.
- LazyGraphRAG (Edge et al., Microsoft, Nov 2024): 0.1% do custo de indexação do GraphRAG, 95/96 statistical wins. Limitações: sem features conversation-specific.
- GraphRAG-Bench (Xiang et al., ICLR 2026): benchmark cross-framework mostrando que GraphRAG frequentemente underperforms vanilla RAG em muitas tarefas reais.

**Sugestões de estudo:**
- Ler o abstract + seção de método do paper Microsoft GraphRAG (arXiv:2404.16130)
- Ler sobre LightRAG para entender a alternativa "leve"
- Pesquisar "GraphRAG limitations" para entender as críticas
- O paper GraphRAG-Bench é importante por mostrar QUANDO grafos ajudam e quando não

### 3.2 Sistemas de Memória Conversacional
**O que cobrir:**
- Graphiti/Zep (Rasmussen et al., 2025): o sistema mais maduro para conversation KGs. Modelo bi-temporal com 4 timestamps, Episode subgraph + Entity subgraph + Community subgraph. 22,800+ GitHub stars. Limitações: projetado para memória de assistente pessoal, não para análise de domínio histórico.
- SGMem (Wu et al., 2025): sentence-level graphs para memória conversacional de longo prazo.
- MAGMA (Jiang et al., 2026): grafos ortogonais (semântico, temporal, causal, entity) por item de memória, +45.5% reasoning accuracy.
- "Does Memory Need Graphs?" (Hu et al., 2026): framework unificado mostrando que sentences-as-nodes pode superar entity-level KG para memória conversacional.

**Sugestões de estudo:**
- Ler o paper do Graphiti (arXiv:2501.13956) — é a referência mais próxima do nosso trabalho
- Entender o modelo bi-temporal (valid_at, invalid_at, created_at, expired_at)
- Pesquisar "conversational knowledge graph" para ver o estado da arte

### 3.3 Extração de Conhecimento de Diálogos
**O que cobrir:**
- Rook (ISWC 2025): formalização de OKE-D (Open Knowledge Extraction from Dialogue). Identifica 4 desafios: cross-turn assertions, anaphora density, dialogue acts, multiple perspectives. A referência acadêmica mais relevante para nosso trabalho.
- LoCoMo (Maharana et al., ACL 2024): benchmark de memória conversacional de longo prazo. Human F1 = 87.9% vs GPT-3.5 = 37.8%. Finding crítico: "RAG performs well when dialogues are transformed into assertions."
- KGGen (Mo et al., NeurIPS 2025): multi-stage extraction com +18% sobre GraphRAG.
- SAC-KG (Chen et al., ACL 2024): Generator→Verifier→Pruner, 89.32% precision em 1M+ nodes.
- CORE-KG: ablation mostrando que remover coreference resolution causa +28.32% node duplication.

**Sugestões de estudo:**
- O paper da Rook (CEUR Vol-4085) é curto e altamente relevante — ler inteiro
- O LoCoMo benchmark valida diretamente nossa abordagem — ler pelo menos o abstract e findings
- Pesquisar "multi-pass knowledge extraction LLM" para ver o estado da arte

### 3.4 Reificação de Claims e Argumentação
**O que cobrir:**
- AIF (Argument Interchange Format): I-Nodes (claims), S-Nodes (schemes), L-Nodes (speech acts)
- DialAM-2024 (ACL 2024): primeiro shared task em dialogical argument mining
- Joel Chan's Discourse Graphs (arXiv:2407.20666): claims linked to evidence via typed edges
- Kialo: 18,000+ debates, 720,000+ claims em produção — valida que claim graphs escalam
- Neo4j Intermediate Node Pattern: a base técnica para reificação em property graphs

**Sugestões de estudo:**
- Pesquisar "argument mining" e "claim detection NLP"
- Entender o conceito de reificação em grafos (transformar relação em nó)
- O Kialo é um bom exemplo prático — explorar o site para ver claims em ação

### 3.5 Tabela Comparativa (nosso sistema vs. existentes)
**O que cobrir:**
- Tabela comparando: Custom Pipeline vs. Microsoft GraphRAG vs. LightRAG vs. Graphiti vs. neo4j-graphrag
- Dimensões: claim reification, temporal modeling, schema control, extraction quality, cost, incremental updates, community detection, query flexibility

**Sugestões de estudo:**
- A tabela já existe na base_research (seção 7.1 do doc 1) — adaptar para o TCC
- Focar nas dimensões onde nosso sistema é SUPERIOR (claim reification, schema control, cost)

---

## Referências-Chave
- Edge et al., 2024 — Microsoft GraphRAG (arXiv:2404.16130)
- Guo et al., 2024 — LightRAG
- Rasmussen et al., 2025 — Graphiti/Zep (arXiv:2501.13956)
- Rook, ISWC 2025 — OKE-D (CEUR Vol-4085)
- Maharana et al., ACL 2024 — LoCoMo (arXiv:2402.17753)
- Mo et al., NeurIPS 2025 — KGGen
- Chen et al., ACL 2024 — SAC-KG
- Xiang et al., ICLR 2026 — GraphRAG-Bench
