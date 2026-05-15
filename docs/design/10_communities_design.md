# Design de Community Detection — Leiden + Summarization

## O que são Communities

Communities (comunidades) são grupos de entidades densamente conectadas no grafo. Elas representam **temas** — clusters de pessoas, organizações, tópicos, e eventos que aparecem juntos frequentemente nas aulas do Jiang.

Exemplos de comunidades que esperamos encontrar:
- "Guerras Púnicas": Roma, Cartago, Aníbal, Cipião, Batalha de Canas, Mediterrâneo
- "Conquistas de Alexandre": Alexandre, Filipe II, Macedônia, Pérsia, Batalha de Gaugamela
- "Queda da República Romana": César, Pompeu, Senado, Guerra Civil, Rubicão

## Por que Communities importam para o projeto

1. **Queries temáticas:** "Quais são os principais temas das aulas?" → retorna community summaries
2. **Contexto para retrieval:** Quando o usuário pergunta sobre Alexandre, podemos incluir o summary da comunidade dele como contexto adicional
3. **Navegação:** Permitem explorar o grafo por tema, não apenas por entidade individual
4. **Validação:** Se uma comunidade não faz sentido temático, algo está errado na extração

## O Algoritmo de Leiden

Leiden é uma evolução do algoritmo de Louvain para community detection. Ele:
1. Otimiza **modularidade** — uma métrica que mede quão densas são as conexões dentro de comunidades vs. entre comunidades
2. Produz **hierarquia** — comunidades em múltiplos níveis (fine-grained no nível 0, broad no nível mais alto)
3. Garante **conectividade** — diferente do Louvain, Leiden garante que cada comunidade é um subgrafo conectado

### Por que Leiden e não Louvain

Louvain pode produzir comunidades desconectadas (um bug conhecido). Leiden corrige isso e geralmente produz partições de qualidade igual ou superior. É o padrão usado pelo Microsoft GraphRAG.

### Hierarquia

Com `includeIntermediateCommunities: true`, Leiden retorna comunidades em múltiplos níveis:
- **Nível 0:** Clusters pequenos e específicos (5-20 entidades)
- **Nível 1:** Clusters médios (20-50 entidades)
- **Nível 2+:** Clusters amplos (50+ entidades)

Para queries temáticas, usamos o nível apropriado:
- "Quais são os temas gerais?" → nível alto
- "O que está relacionado a Alexandre?" → nível baixo

## Summarization com LLM

Após detectar comunidades, geramos um **summary** para cada uma usando um LLM:
1. Coletamos todos os membros da comunidade (nomes, descrições, relações internas)
2. Passamos ao LLM: "Resuma esta comunidade de entidades de uma aula de história"
3. O LLM retorna um parágrafo descrevendo o tema, figuras-chave, e por que se agrupam
4. Armazenamos o summary + embedding no nó Community

Esses summaries são o que o retrieval usa para responder queries temáticas.

## Incrementalidade

Quando um novo episódio é adicionado:
1. Novas entidades e relações entram no grafo
2. Re-rodamos Leiden com `seedProperty` (usa comunidades anteriores como ponto de partida)
3. Apenas comunidades que MUDARAM precisam de re-summarization
4. Para 1 episódio novo por semana: ~5-15 comunidades re-sumarizadas (não todas)

## Dependência: Neo4j GDS

Community detection requer o plugin **Graph Data Science (GDS)** do Neo4j. Ele:
- Projeta um grafo in-memory (subconjunto do grafo completo)
- Roda algoritmos sobre esse grafo projetado
- Escreve resultados de volta nos nós

GDS está incluído no nosso `docker-compose.yml` via `NEO4J_PLUGINS=["graph-data-science"]`.

## Notas para o TCC

### O que documentar no artigo:
1. **O conceito de community detection** — o que são comunidades em grafos
2. **Leiden vs. Louvain** — por que Leiden é preferido (conectividade garantida)
3. **Hierarquia** — múltiplos níveis de granularidade
4. **Summarization** — como LLM transforma clusters em descrições legíveis
5. **O papel no retrieval** — queries temáticas usam community summaries
6. **Incrementalidade** — seeded Leiden para atualizações eficientes
7. **Referências:** Microsoft GraphRAG (arXiv:2404.16130), DF Leiden (arXiv:2405.11658)
