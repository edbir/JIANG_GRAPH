# TCC Outline — Detailed Section-by-Section Plan

## Title (working)
**GraphRAG para Dados Conversacionais de Aulas de História: Uma Arquitetura Custom de Knowledge Graph e Retrieval com Extração Multi-Pass**

**English:** *GraphRAG for Conversational History Class Data: A Custom Knowledge Graph and Retrieval Architecture with Multi-Pass Extraction*

---

## Target Length: ~50–60 pages (25,000–30,000 words)

---

## Section Breakdown

### 1. INTRODUÇÃO (4–5 pages, ~2,500 words)

| Subseção | Palavras | O que contém |
|----------|----------|--------------|
| 1.1 Contextualização | ~600 | Crescimento de vídeo educacional, dados não-estruturados, dificuldade de busca em horas de conteúdo |
| 1.2 RAG e Limitações | ~400 | RAG tradicional perde causalidade, relações, estrutura argumentativa |
| 1.3 GraphRAG como Evolução | ~400 | KGs preservam estrutura, paper Microsoft GraphRAG como marco |
| 1.4 O Domínio | ~300 | Predictive History, Prof. Jiang, tipo de conteúdo, volume |
| 1.5 Limitações dos Frameworks | ~300 | Nenhum framework cobre claims + temporal + domínio histórico |
| 1.6 Objetivo e Contribuições | ~300 | Objetivo claro + 4 contribuições listadas |
| 1.7 Organização do Documento | ~200 | Parágrafo descrevendo cada seção |

**Referências-chave:** Lewis et al. 2020 (RAG), Edge et al. 2024 (GraphRAG), LoCoMo ACL 2024, openpraxis.org video growth

---

### 2. CONCEITOS E TECNOLOGIAS (8–10 pages, ~5,000 words)

| Subseção | Palavras | O que contém |
|----------|----------|--------------|
| 2.1 RAG | ~700 | Problema de hallucination, pipeline embed→search→generate, chunking, limitações |
| 2.2 Knowledge Graphs | ~800 | Definição, triplas, RDF vs Property Graph, ontologia, reificação |
| 2.3 GraphRAG | ~700 | Combinação KG+RAG, Microsoft GraphRAG, hybrid retrieval, community detection |
| 2.4 Neo4j e Cypher | ~600 | Property graph DB, Cypher, vector indexes, GDS, MERGE |
| 2.5 Extração de Conhecimento com LLMs | ~800 | Single vs multi-pass, structured outputs, tiered models, entity resolution |
| 2.6 Community Detection (Leiden) | ~600 | O que é, Louvain→Leiden, hierarquia, gamma, incrementalidade |
| 2.7 Model Context Protocol (MCP) | ~500 | O que é, Tools/Resources/Prompts, por que usar |

**Referências-chave:** Lewis et al. 2020, ISO GQL 39075:2024, Edge et al. 2024, Traag et al. 2019, KGGen NeurIPS 2025, Anthropic MCP spec

---

### 3. TRABALHOS RELACIONADOS (5–7 pages, ~3,500 words)

| Subseção | Palavras | O que contém |
|----------|----------|--------------|
| 3.1 Sistemas GraphRAG | ~800 | Microsoft GraphRAG, LightRAG, LazyGraphRAG, GraphRAG-Bench |
| 3.2 Memória Conversacional | ~700 | Graphiti/Zep, SGMem, MAGMA, "Does Memory Need Graphs?" |
| 3.3 Extração de Diálogos | ~700 | OKE-D (Rook), LoCoMo, KGGen, SAC-KG, CORE-KG |
| 3.4 Reificação de Claims | ~500 | AIF, DialAM-2024, Discourse Graphs, Kialo |
| 3.5 Tabela Comparativa | ~500 | Custom vs GraphRAG vs LightRAG vs Graphiti vs neo4j-graphrag (8 dimensões) |
| 3.6 Posicionamento | ~300 | Onde nosso trabalho se encaixa, o gap que preenche |

**Referências-chave:** Edge et al. 2024, Rasmussen et al. 2025, Rook ISWC 2025, Maharana ACL 2024, Mo et al. NeurIPS 2025

---

### 4. DESENVOLVIMENTO (15–18 pages, ~9,000 words)

| Subseção | Palavras | O que contém |
|----------|----------|--------------|
| 4.1 Visão Geral da Arquitetura | ~800 | Diagrama 4 camadas, fluxo E2E, decisão custom vs framework |
| 4.2 Schema do Knowledge Graph | ~1,200 | 3 princípios, 9 node types, relationships, constraints, escala |
| 4.3 Preprocessing | ~900 | Text cleaning, two-pass chunking, overlap, parâmetros |
| 4.4 Extração Multi-Pass | ~1,500 | 5 passes em detalhe, structured outputs, prompts, caching, tiered models |
| 4.5 Entity Resolution | ~800 | Cascata 4 estágios, thresholds, princípio conservador, merge map |
| 4.6 Upsert no Grafo | ~700 | MERGE patterns, ordem de inserção, enrichment, IDs determinísticos |
| 4.7 Community Detection | ~700 | Leiden no GDS, hierarquia, summarization via LLM, incrementalidade |
| 4.8 Retrieval | ~1,200 | 4 tipos de query, vector+graph fan-out, T2C, RRF, context assembly |
| 4.9 MCP Server | ~600 | 6 tools, design principles, separação server/client |
| 4.10 Custo e Otimizações | ~500 | Prompt caching, batch API, tiered models, custo por episódio |

**Conteúdo técnico:** Diagramas, trechos de código, exemplos de Cypher, tabela de custos

---

### 5. AVALIAÇÃO (5–7 pages, ~3,500 words)

| Subseção | Palavras | O que contém |
|----------|----------|--------------|
| 5.1 Metodologia | ~500 | O que avaliar, como medir, baseline (RAG tradicional) |
| 5.2 Qualidade de Extração | ~800 | Precision/recall de entidades, claims, relações (sample manual) |
| 5.3 Qualidade do Grafo | ~500 | Métricas estruturais (orphans, duplicatas, conectividade) |
| 5.4 Qualidade do Retrieval | ~800 | Comparação GraphRAG vs vector-only em 4 tipos de query |
| 5.5 Custo e Performance | ~400 | Custo real, latência, throughput |
| 5.6 Discussão | ~500 | Onde funciona bem, onde falha, trade-offs |

**⚠️ PRECISA DE DADOS REAIS** — esta seção não pode ser escrita agora

---

### 6. CONCLUSÃO (2–3 pages, ~1,500 words)

| Subseção | Palavras | O que contém |
|----------|----------|--------------|
| 6.1 Síntese | ~400 | Resumo do que foi feito e dos resultados |
| 6.2 Contribuições | ~400 | Listar as 4 contribuições e como foram realizadas |
| 6.3 Limitações | ~300 | Limitações honestas do trabalho |
| 6.4 Trabalhos Futuros | ~400 | O que pode ser estendido |

**⚠️ PRECISA DE RESULTADOS** — só pode ser finalizado depois da avaliação

---

## REFERÊNCIAS (~2 pages)
~40–60 referências esperadas, cobrindo:
- Fundamentos (RAG, KGs, Leiden)
- GraphRAG papers (Microsoft, LightRAG, LazyGraphRAG)
- Extração (KGGen, SAC-KG, ODKE+, CORE-KG)
- Conversational (LoCoMo, OKE-D, Graphiti, SGMem)
- Tecnologias (Neo4j, MCP, OpenAI)
- Vídeo educacional (openpraxis, Noetel, Mayer)

---

## Total Estimado
| Seção | Páginas | Palavras |
|-------|---------|----------|
| 1. Introdução | 4–5 | 2,500 |
| 2. Conceitos | 8–10 | 5,000 |
| 3. Trabalhos Relacionados | 5–7 | 3,500 |
| 4. Desenvolvimento | 15–18 | 9,000 |
| 5. Avaliação | 5–7 | 3,500 |
| 6. Conclusão | 2–3 | 1,500 |
| Referências | 2 | — |
| **TOTAL** | **~45–55** | **~25,000** |

---

## O que pode ser escrito AGORA (sem dados de avaliação)
- ✅ Seção 1 (Introdução) — quase toda, exceto menção a resultados
- ✅ Seção 2 (Conceitos e Tecnologias) — 100% factual
- ✅ Seção 3 (Trabalhos Relacionados) — 100% factual
- ✅ Seção 4 (Desenvolvimento) — 100% factual (descreve a implementação)
- ❌ Seção 5 (Avaliação) — precisa de dados reais
- ❌ Seção 6 (Conclusão) — precisa dos resultados da avaliação
