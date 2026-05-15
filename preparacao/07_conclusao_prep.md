# Preparação — Seção 6: Conclusão

## Objetivo da Seção
Fechar o trabalho: resumir o que foi feito, o que foi aprendido, quais são as contribuições, e o que pode ser feito no futuro. Deve ser concisa (1-2 páginas) e não introduzir informação nova.

---

## Estrutura Planejada

### 6.1 Resumo do Trabalho
**O que cobrir:**
- Reafirmar o problema (conteúdo educacional em áudio é difícil de consultar)
- Reafirmar a solução (GraphRAG custom com ontologia específica)
- Resumir o que foi construído (pipeline completo: ingestion + retrieval + MCP)
- Principais resultados quantitativos (da seção de avaliação)

**Sugestões de estudo:**
- Ler conclusões de outros TCCs da UFRGS para ver o padrão de escrita
- Deve ser possível ler APENAS a introdução + conclusão e entender o trabalho

### 6.2 Contribuições
**O que cobrir:**
- Lista clara das contribuições do trabalho:
  1. Uma ontologia específica para dados conversacionais de aulas de história (claims como nós, bi-temporal)
  2. Um pipeline de extração multi-pass com structured outputs e entity resolution em cascata
  3. Uma camada de retrieval híbrida adaptativa (4 estratégias por tipo de query)
  4. Avaliação comparativa demonstrando superioridade sobre RAG tradicional para queries estruturais
  5. (Se aplicável) O sistema como ferramenta educacional — permite consultar o conhecimento de um professor

**Sugestões de estudo:**
- Contribuições devem ser ESPECÍFICAS e VERIFICÁVEIS
- Cada contribuição deve ter evidência na seção de avaliação

### 6.3 Trabalhos Futuros
**O que cobrir:**
- Melhorias no pipeline:
  - LLM-based segmentation com embeddings (v2 do chunker)
  - Cross-encoder reranking no retrieval (Stage 2 após RRF)
  - Fine-tuning de modelos menores para extração (reduzir custo)
- Extensões do sistema:
  - Suporte a múltiplos speakers (podcast com convidados)
  - Suporte a outros idiomas (português, etc.)
  - Interface web para exploração visual do grafo
  - Verificação automática de previsões (conectar a fontes de notícias)
- Aplicações:
  - Pedagogical agents baseados no knowledge graph
  - Aplicação a outros domínios (podcasts de ciência, direito, medicina)
  - Integração com plataformas educacionais

**Sugestões de estudo:**
- Trabalhos futuros devem ser REALISTAS (coisas que de fato poderiam ser feitas)
- Mencionar o campo de "AI-generated pedagogical agents" como extensão natural
- Não prometer o que não foi validado

### 6.4 Considerações Finais
**O que cobrir:**
- 1-2 parágrafos de reflexão pessoal sobre o projeto
- O que foi mais desafiador
- O que foi mais surpreendente nos resultados
- A relevância do trabalho no contexto atual (LLMs + knowledge graphs como tendência)

**Sugestões de estudo:**
- Ser genuíno — a banca percebe quando é genérico
- Conectar com o momento atual da IA (2025-2026: era dos agentes, MCP, GraphRAG)

---

## Dica para a escrita
A conclusão NÃO é um resumo do artigo inteiro. É uma síntese: o que aprendemos, o que contribuímos, e para onde isso vai. Deve deixar o leitor com uma impressão clara do valor do trabalho.
