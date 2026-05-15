# 2. CONCEITOS E TECNOLOGIAS

Esta seção apresenta os conceitos teóricos e tecnologias necessários para compreender a arquitetura proposta neste trabalho.

## 2.1 Retrieval-Augmented Generation (RAG)

Large Language Models (LLMs) possuem uma limitação fundamental: seu conhecimento é estático, limitado aos dados de treinamento. Além disso, LLMs frequentemente produzem respostas factualmente incorretas — fenômeno conhecido como "alucinação" (Huang et al., 2023). Retrieval-Augmented Generation (RAG) é uma técnica que endereça ambos os problemas ao combinar um componente de recuperação de informação com o modelo generativo (Lewis et al., 2020).

O pipeline RAG típico funciona em três etapas:

1. **Indexação:** documentos são divididos em fragmentos (chunks) de tamanho fixo (~500–1500 tokens) e convertidos em vetores numéricos de alta dimensão (embeddings) via modelos como text-embedding-3-small da OpenAI. Esses vetores são armazenados em um banco vetorial.

2. **Recuperação:** quando o usuário faz uma pergunta, esta é também convertida em um embedding. Busca-se no banco vetorial os chunks cujos embeddings são mais similares ao da pergunta (tipicamente via similaridade de cosseno).

3. **Geração:** os chunks recuperados são inseridos no prompt do LLM como contexto, e o modelo gera uma resposta baseada nessa informação.

O RAG representa um avanço significativo sobre LLMs puros: permite acesso a informação atualizada sem re-treinamento, fundamenta respostas em fontes verificáveis, e reduz alucinações ao fornecer contexto factual. No entanto, a abordagem vetorial apresenta limitações estruturais. Chunks de texto são tratados como unidades independentes e planas — toda informação relacional entre entidades, causalidade entre eventos, e estrutura argumentativa é perdida no processo de fragmentação. Para domínios onde relacionamentos e estrutura são centrais à compreensão — como história, onde eventos causam outros eventos e figuras interagem em redes complexas — essas limitações se tornam especialmente problemáticas.

## 2.2 Knowledge Graphs

Um Knowledge Graph (KG) é uma estrutura de dados que representa informação na forma de um grafo, onde nós representam entidades (pessoas, lugares, conceitos) e arestas representam relações entre elas (Hogan et al., 2021). A unidade básica de informação é a tripla: (sujeito)–[predicado]→(objeto). Por exemplo: (Alexandre, o Grande)–[CONQUISTOU]→(Império Persa).

Existem dois paradigmas principais para representação de grafos de conhecimento:

**RDF (Resource Description Framework):** padrão W3C baseado em triplas sujeito-predicado-objeto. Utiliza URIs para identificação global, ontologias formais (OWL) para raciocínio lógico, e SPARQL como linguagem de consulta. É ideal para interoperabilidade e dados linkados (Linked Data), mas requer mecanismos complexos (blank nodes, reificação via nomeação) para atribuir propriedades a relações.

**Property Graphs:** modelo onde tanto nós quanto arestas podem possuir propriedades arbitrárias (pares chave-valor). A linguagem de consulta predominante é Cypher, baseada em pattern matching visual. Cypher foi padronizado internacionalmente como GQL (ISO/IEC 39075:2024) — a primeira nova linguagem de banco de dados ISO desde SQL em 1987.

Este trabalho adota o modelo de property graph por três razões: (1) flexibilidade de schema — propriedades podem ser adicionadas a nós e arestas sem reestruturação; (2) expressividade de consulta — Cypher é mais natural para traversal multi-hop que SPARQL; (3) alinhamento com o ecossistema Neo4j, que oferece vector indexes nativos, community detection via GDS, e integração madura com LLMs.

**Ontologia** refere-se ao schema que define quais tipos de entidades e relações são permitidos no grafo. Uma ontologia bem definida funciona como contrato: guia a extração (o LLM sabe o que procurar), previne inconsistências (tipos inválidos são rejeitados), e habilita queries previsíveis (o consumidor sabe o que existe no grafo).

**Reificação** é o padrão de transformar uma relação em um nó intermediário para permitir que ela possua suas próprias propriedades e relações. Em vez de modelar uma afirmação como propriedade de uma aresta entre duas entidades, cria-se um nó Claim que conecta-se ao utterance de origem, ao episódio em que foi dita, e às entidades que referencia. O Neo4j documenta este padrão como "Intermediate Node Pattern" em seu curso oficial de modelagem de dados.

## 2.3 GraphRAG

GraphRAG combina Knowledge Graphs com Retrieval-Augmented Generation, utilizando a estrutura do grafo para enriquecer o contexto fornecido ao LLM além do que busca vetorial pura oferece.

O trabalho seminal da Microsoft (Edge et al., 2024) estabeleceu o termo e demonstrou sua eficácia em tarefas de *sensemaking* — compreensão global de grandes corpora. A arquitetura Microsoft GraphRAG opera em duas fases:

**Indexação:** textos são processados por um LLM que extrai entidades e relações, construindo um grafo de conhecimento. Sobre esse grafo, o algoritmo de Leiden detecta comunidades (clusters de entidades densamente conectadas). Cada comunidade recebe um sumário gerado por LLM em múltiplos níveis de hierarquia.

**Consulta:** queries "locais" (sobre entidades específicas) utilizam entity search + neighborhood expansion. Queries "globais" (sobre temas gerais) sintetizam os sumários das comunidades relevantes. 

Abordagens GraphRAG superam RAG vetorial em 70–80% das tarefas complexas de sensemaking (Edge et al., 2024). No entanto, a implementação da Microsoft possui limitações: reconhece apenas 4 tipos de entidade (Person, Organization, Geography, Event); consome ~610.000 tokens por query global; exige reindexação completa para atualizações; e não possui otimizações para dados conversacionais.

Alternativas relevantes incluem **LightRAG** (Guo et al., 2024), que usa extração aberta com ~100 tokens por query (6.000× mais eficiente), e **LazyGraphRAG** (Edge et al., 2024b), que elimina sumarização upfront com custo de indexação 0.1% do GraphRAG completo, alcançando 95/96 vitórias estatísticas contra todas as alternativas.

O conceito de **retrieval híbrido** — combinar busca vetorial (semântica) com traversal de grafo (estrutural) — é central a este trabalho. Busca vetorial encontra pontos de entrada relevantes; traversal de grafo expande o contexto com relações, claims, e entidades conectadas que pura similaridade semântica não capturaria.

## 2.4 Neo4j e Cypher

Neo4j é o banco de dados de grafos mais utilizado globalmente, implementando o modelo de property graph. Sua relevância para este trabalho se deve a funcionalidades específicas:

**Cypher** é a linguagem de consulta de Neo4j, baseada em pattern matching visual onde padrões de grafo são expressos com ASCII art: `(a:Person)-[:KNOWS]->(b:Person)`. A versão 5.x introduziu suporte a queries vetoriais nativas, eliminando a necessidade de um banco vetorial separado.

**Vector Indexes** permitem busca por similaridade semântica diretamente no Neo4j. Cada nó pode armazenar um vetor de embedding e ser consultado via approximate nearest neighbor (ANN):

```cypher
CALL db.index.vector.queryNodes('utterance_embeddings', 5, $queryEmbedding)
YIELD node AS chunk, score
```

**MERGE** é o operador de upsert idempotente: cria um nó/relação se não existe, ou encontra o existente se já foi criado. Com cláusulas `ON CREATE SET` e `ON MATCH SET`, permite enriquecimento progressivo sem duplicação — essencial para um pipeline de ingestão incremental.

**Neo4j GDS (Graph Data Science)** é uma biblioteca de algoritmos de grafo que inclui community detection (Leiden, Louvain), centralidade (PageRank, Betweenness), e similarity. Para este trabalho, o algoritmo de Leiden detecta comunidades temáticas no grafo de entidades.

## 2.5 Extração de Conhecimento com LLMs

A construção de Knowledge Graphs a partir de texto não-estruturado tradicionalmente dependia de pipelines de NLP (NER + Relation Extraction + Coreference Resolution). Com o advento de LLMs, tornou-se possível realizar extração de ponta a ponta via prompting — alimentando texto e instruções, o modelo retorna entidades e relações estruturadas.

**Single-pass vs. Multi-pass.** Solicitar ao LLM que extraia entidades, relações, claims e resolva correferências em uma única chamada produz resultados subótimos — o modelo faz tudo superficialmente. Pesquisas convergem na superioridade da extração multi-pass: ECR demonstra +16% em F1 de triplas (single vs. multi-pass); KGGen (Mo et al., NeurIPS 2025) alcança 18% de melhoria sobre GraphRAG via extração multi-estágio (extract→cluster→normalize); SAC-KG (Chen et al., ACL 2024) atinge 89.32% de precisão com pipeline Generator→Verifier→Pruner em escala de 1M+ nós.

**Structured Outputs** são um mecanismo de constrained decoding oferecido por provedores de LLM (OpenAI Structured Outputs, agosto 2024) que garante 100% de conformidade com um JSON Schema fornecido. Diferente de pedir ao modelo que "retorne JSON" (conformidade < 40% com GPT-4-0613), structured outputs utilizam gramáticas restritivas durante a decodificação, tornando impossível gerar output fora do schema. Benchmarks (JSONSchemaBench, ICML 2025) demonstram que constrained decoding não apenas garante formato, mas melhora qualidade downstream em até 4%.

**Tiered Model Selection** é a estratégia de usar modelos diferentes por complexidade de tarefa: modelos pequenos e baratos (GPT-4o-mini) para tarefas de alto volume e baixa ambiguidade (extração de entidades), e modelos maiores (GPT-4o) para tarefas que requerem raciocínio nuançado (detecção de stance, resolução de correferência). Isso otimiza a relação custo/qualidade do pipeline.

**Entity Resolution** é o processo de identificar quando nomes diferentes referem-se à mesma entidade real ("Alexandre", "Alexandre, o Grande", "Alexander III of Macedon"). Abordagens modernas usam cascatas que vão do barato ao caro: normalização de texto (40% dos duplicatas), fuzzy string matching via token_sort_ratio (35%), similaridade de embeddings (20%), e julgamento LLM apenas para os 5% restantes genuinamente ambíguos. CORE-KG demonstrou que omitir resolução de correferência causa aumento de 28.32% em duplicação de nós.

## 2.6 Community Detection e o Algoritmo de Leiden

Community detection em grafos identifica conjuntos de nós que são mais densamente conectados entre si do que com o restante do grafo. Intuitivamente, comunidades correspondem a clusters temáticos — num grafo de história, uma comunidade poderia agrupar "Império Romano", "Júlio César", "Senado Romano", e "Guerra Civil Romana".

O **algoritmo de Louvain** (Blondel et al., 2008) é o método clássico: otimiza modularidade movendo nós entre comunidades iterativamente. No entanto, Louvain pode produzir comunidades mal-conectadas (internamente desconexas).

O **algoritmo de Leiden** (Traag et al., 2019) resolve essa garantia: além de otimizar modularidade, garante que todas as comunidades detectadas são internamente bem-conectadas. Leiden adiciona uma fase de refinamento que Louvain não possui, resultando em partições de qualidade superior.

**Comunidades hierárquicas** são obtidas executando Leiden com `includeIntermediateCommunities: true`, produzindo partições em múltiplos níveis de resolução: Level 0 (granular — "Campanhas de Alexandre na Pérsia"), Level 1 (médio — "Conquistas Helênicas"), Level 2 (amplo — "Antiguidade Clássica"). O parâmetro gamma controla a resolução: valores maiores produzem comunidades menores e mais numerosas.

Para **atualizações incrementais**, o mecanismo de seeded Leiden permite reutilizar a partição anterior como ponto de partida. Quando um novo episódio adiciona entidades ao grafo, apenas as comunidades afetadas são reparticionadas — evitando recomputar todo o grafo. Dynamic Frontier Leiden (Sahu, 2024) formaliza este processo, alcançando modularidade comparável ao Leiden estático com runtime dramaticamente reduzido.

No contexto de GraphRAG, comunidades servem como base para respostas temáticas: cada comunidade recebe um sumário gerado por LLM a partir de seus membros e relações, permitindo responder perguntas como "quais são os principais temas do curso?" sem varrer todos os nós.

## 2.7 Model Context Protocol (MCP)

O Model Context Protocol (MCP) é um padrão aberto para conectar sistemas de IA a fontes de dados externas. Anunciado pela Anthropic em novembro de 2024 e posteriormente doado à Agentic AI Foundation sob a Linux Foundation (dezembro de 2025, co-fundada com Block e OpenAI), o protocolo atingiu 97 milhões de downloads mensais de SDK e 10.000+ servidores até final de 2025.

MCP opera sobre JSON-RPC 2.0 e define três primitivos:

- **Tools:** funções que o LLM pode invocar (ex: buscar no grafo, comparar entidades)
- **Resources:** dados estáticos que o LLM pode ler (ex: o schema do grafo)
- **Prompts:** templates reutilizáveis para interações comuns

A vantagem do MCP sobre APIs REST tradicionais é que o LLM client-side pode autonomamente decidir qual tool invocar com base na pergunta do usuário. Se o usuário pergunta "o que Jiang disse sobre Roma?", o LLM escolhe `search_lectures`; se pergunta "compare César e Alexandre", escolhe `compare_entities`. O sistema torna-se auto-navegável sem UI dedicada.

Para este trabalho, um servidor MCP custom expõe 6 tools que encapsulam as 4 estratégias de retrieval (vetorial, templates Cypher, Text-to-Cypher, community summaries). O LLM client (Claude, GPT, ou qualquer cliente MCP-compatível) atua como interface final com o usuário — consulta o grafo, recebe contexto estruturado, e gera respostas em linguagem natural.
