# 3. TRABALHOS RELACIONADOS

Esta seção posiciona o presente trabalho no contexto da literatura existente, identificando as lacunas que motivam a construção de um pipeline custom.

## 3.1 Sistemas GraphRAG Existentes

**Microsoft GraphRAG** (Edge et al., 2024) é o trabalho seminal que formalizou a combinação de knowledge graphs com RAG para tarefas de sensemaking. O sistema extrai entidades e relações via LLM, aplica community detection hierárquica (Leiden), e gera sumários por comunidade. Dois modos de consulta — local (entity-centric) e global (community summaries) — cobrem queries pontuais e temáticas respectivamente.

Apesar de sua contribuição conceitual, o Microsoft GraphRAG apresenta limitações práticas significativas. Sua extração default reconhece apenas quatro tipos de entidade (Person, Organization, Geography, Event), insuficientes para capturar stances, previsões e análises causais presentes em aulas de história. O modo de query global consome aproximadamente 610.000 tokens por consulta — versus ~100 tokens para LightRAG, uma diferença de 6.000×. O repositório oficial adverte: "Using GraphRAG with your data out of the box may not yield the best possible results." Shereshevsky (2025) relata custos de indexação de $33K para datasets grandes, inviáveis para um corpus de 134 episódios que requer atualizações incrementais.

**LightRAG** (Guo et al., HKU, 2024) propõe uma alternativa leve: extração aberta sem schema constraints, com atualizações incrementais via graph union e queries eficientes (~100 tokens). Sua simplicidade tem custo: zero schema constraints produzem tipos de relação inconsistentes e duplicação de entidades. O framework de análise Plumber demonstra que extração aberta resulta em 40% de erro de predicados. O paper foi retirado (withdrawn) da revisão no ICLR 2025.

**LazyGraphRAG** (Edge et al., Microsoft Research, novembro 2024) elimina a sumarização upfront, com custo de indexação a 0.1% do GraphRAG completo. Em avaliação, alcançou 95 de 96 vitórias estatísticas contra Vector RAG, GraphRAG Local/Global/DRIFT, LightRAG, RAPTOR, e TREX. É a recomendação padrão para corpora documentais não-conversacionais. No entanto, não possui features para dados de diálogo: sem speaker attribution, sem modelagem temporal, sem reificação de claims.

**GraphRAG-Bench** (Xiang et al., ICLR 2026) fornece o primeiro benchmark rigoroso cross-framework, com a conclusão de que GraphRAG frequentemente underperforms vanilla RAG em tarefas reais, oferecendo guidelines sobre quando estrutura de grafo fornece benefícios mensuráveis — tipicamente em queries multi-hop, comparativas, e de síntese global.

## 3.2 Sistemas de Memória Conversacional

**Graphiti/Zep** (Rasmussen et al., 2025; arXiv:2501.13956) é o sistema mais maduro para knowledge graphs conversacionais. Com 22.800+ estrelas no GitHub e 25.000+ downloads semanais no PyPI, define o grafo como G = (N, E, φ) com três subgrafos: Episode (preserva dados raw), Entity (conhecimento semântico extraído), e Community (sumários temáticos).

A inovação central do Graphiti é seu modelo bi-temporal: cada aresta semântica carrega quatro timestamps — `created_at` (quando o fato foi ingerido), `expired_at` (quando foi invalidado no sistema), `valid_at` (quando o fato tornou-se verdadeiro na realidade), e `invalid_at` (quando o fato deixou de ser verdadeiro). Quando informação contraditória chega, a aresta antiga é invalidada e o texto do fato é regenerado para refletir a evolução.

Apesar de sua sofisticação temporal, Graphiti é projetado para memória de assistente pessoal — rastrear preferências de usuário, fatos pessoais, conversas anteriores. Sua extração de entidades é genérica, sem ontologia de domínio. Para análise histórica específica (figuras, impérios, batalhas, relações causais entre eventos), é necessário controle fino sobre tipos de entidade e relação.

**SGMem** (Wu et al., 2025; arXiv:2509.21212) representa diálogo como grafos sentence-level dentro de unidades chunked, capturando associações em nível de turno, rodada, e sessão. Supera baselines em LongMemEval e LoCoMo.

**MAGMA** (Jiang et al., 2026; arXiv:2601.03236) estende o paradigma com grafos ortogonais (semântico, temporal, causal, e de entidades) por item de memória, alcançando 45.5% mais acurácia em raciocínio via traversal guiado por política.

**"Does Memory Need Graphs?"** (Hu et al., 2026; arXiv:2601.01280) introduz um framework unificado decompondo sistemas de memória de diálogo em componentes core. Seu achado principal: sentences-as-nodes pode superar abordagens entity-level KG para memória conversacional, fornecendo orientação sobre representação, organização, manutenção, e retrieval.

## 3.3 Extração de Conhecimento de Diálogos

**OKE-D** (Rook, ISWC 2025 Doctoral Consortium; CEUR Vol-4085) é a referência acadêmica mais diretamente relevante para este trabalho. Formaliza Open Knowledge Extraction from Dialogue e identifica quatro desafios estruturais únicos de dados conversacionais: (1) cross-turn assertions com distâncias sujeito-objeto maiores que em documentos; (2) densidade de anáfora e pronomes requerendo resolução robusta de correferência; (3) dialogue acts onde nem todas as falas assertam fatos (algumas são opiniões, sugestões, hedges); (4) múltiplas perspectivas onde falantes expressam visões conflitantes.

Rook propõe uma ontologia perspective-aware integrando identidade do speaker, contexto temporal, e dialogue acts via SIO e PROV-O. O trabalho demonstra que mesmo modelos frontier (Gemini 2.5 Pro) falham em extrair o conhecimento importante de diálogos sem prompting especializado.

**LoCoMo** (Maharana et al., ACL 2024; arXiv:2402.17753) é o benchmark de memória conversacional de longo prazo mais relevante, avaliando sobre ~300 turnos e até 35 sessões. Seus achados quantificam o gap: human F1 = 87.9%, GPT-3.5-16K = 37.8%, melhor RAG = 41.4% — LLMs ficam 56% atrás de humanos. O achado mais crítico para nosso trabalho: "RAG does particularly well when dialogues are transformed into a database of assertions (observations) about each speaker's life and persona" — validação direta da abordagem de extração para knowledge graph.

**KGGen** (Mo et al., NeurIPS 2025) demonstra extração multi-estágio (extract→cluster→normalize) alcançando 66.07% no benchmark MINE versus 47.80% do GraphRAG e 29.84% de OpenIE — uma melhoria de 18%. Claude Sonnet 3.5 alcançou o maior score individual (73%).

**SAC-KG** (Chen et al., ACL 2024) propõe um pipeline Generator→Verifier→Pruner alcançando 89.32% de precisão em escala de 1M+ nós, representando >20% de aumento sobre o estado da arte anterior.

**CORE-KG** demonstra via ablation que remover resolução de correferência causa aumento de 28.32% em duplicação de nós, e que remover prompts estruturados causa 73% mais ruído — evidência quantitativa para cada componente do pipeline.

## 3.4 Reificação de Claims e Frameworks de Argumentação

A modelagem de claims como nós first-class em grafos de conhecimento é fundamentada por múltiplas tradições:

**AIF (Argument Interchange Format)** define I-Nodes (claims/informação), S-Nodes (aplicações de scheme subdividas em inferência, conflito, e preferência), e L-Nodes (speech acts em diálogo). Um grafo AIF é uma estrutura bipartite onde arestas sempre conectam I-Nodes a S-Nodes e vice-versa.

**DialAM-2024** (Ruiz-Dolz et al., ArgMining Workshop at ACL 2024) apresenta o primeiro shared task em dialogical argument mining usando Inference Anchoring Theory, modelando diálogo com locution nodes (o que foi dito), proposition nodes (claims), e relation nodes (relações argumentativas/ilocucionárias).

**Discourse Graphs** (Joel Chan, arXiv:2407.20666, 2024) modelam knowledge claims linkados a evidências via typed edges (supports, opposes, informs), com 3 anos de deployment prático em comunidades de pesquisa distribuídas.

**Kialo** demonstra escalabilidade prática: 18.000+ debates e 720.000+ claims em produção, usando estrutura de árvore onde cada claim é Pro ou Con relativo a seu pai. Prova que grafos de argumentação escalam bem quando claims são nós first-class.

O **Intermediate Node Pattern do Neo4j** é a base técnica para reificação em property graphs: substituir uma relação que precisa de metadados ricos por um nó. Para claims, isso permite anexar confiança, stance, status de verificação, tipo de claim, e relações claim-a-claim (SUPPORTS, CONTRADICTS, REVISES_PREDICTION).

## 3.5 Análise Comparativa

A Tabela 1 compara o sistema proposto com as alternativas existentes em dimensões relevantes para o caso de uso de aulas de história.

| Dimensão | Custom (este trabalho) | Microsoft GraphRAG | LightRAG | Graphiti/Zep |
|----------|----------------------|-------------------|----------|--------------|
| Reificação de claims | Nativa (nós first-class) | Não suporta | Não suporta | Não suporta |
| Modelagem temporal | Bi-temporal (when said + what refers to) | Não suporta | Não suporta | 4 timestamps |
| Controle de schema | Total (ontologia custom) | 4 entity types fixos | Extração aberta | Entidades genéricas |
| Qualidade de extração | Schema-constrained (98.8% ODKE+) | Extração genérica | 40% predicate errors | Extração genérica |
| Custo (134 episódios) | ~$15–45 | ~$33K (estimado) | ~$15–30 | Variável |
| Updates incrementais | MERGE idempotente | Reindex parcial | Graph union | Dinâmico |
| Community detection | Leiden via GDS (hierárquico) | Leiden hierárquico | Nenhum | Label propagation |
| Tokens por query | ~100–1000 | ~610.000 (global) | ~100 | Variável |

## 3.6 Posicionamento

Nenhum sistema existente endereça a interseção de: (1) modelagem de claims como entidades first-class com relações claim-a-claim; (2) extração multi-pass com schema constrained para domínio histórico; (3) tracking temporal de previsões geopolíticas verificáveis; (4) retrieval híbrido adaptativo por tipo de query. 

O presente trabalho preenche este gap construindo um pipeline custom que incorpora os melhores padrões de cada sistema — a arquitetura temporal do Graphiti, o utterance chaining de Di Bratto, o dual-path retrieval do lexical graph pattern da Neo4j, a estrutura argumentativa do AIF, e a sumarização comunitária do Microsoft GraphRAG — adaptados para um domínio de aulas de história onde claims, previsões e relações causais são o conteúdo de maior valor analítico.
