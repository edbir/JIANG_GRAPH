# Test Queries — Prompts para Validação do Sistema

## Propósito

Estes prompts servem para testar o sistema após a ingestão dos primeiros episódios. Estão organizados por tipo de query (factual, comparative, temporal, thematic) e por complexidade (simples → difícil).

Um bom sistema deve:
- Responder queries simples com precisão
- Responder queries complexas com contexto rico do grafo
- Admitir quando não tem informação suficiente
- Citar episódios/momentos específicos como fonte

---

## Factual Queries (Vector + Graph Fan-Out)

### Simples (entidade única, resposta direta)
1. "What did Jiang say about Alexander the Great?"
2. "Tell me about the fall of the Roman Empire according to Jiang."
3. "What does Jiang think about democracy?"
4. "What did Jiang say about the Battle of Thermopylae?"
5. "How does Jiang describe Julius Caesar?"
6. "What is Jiang's view on the Ottoman Empire?"
7. "What did Jiang say about Genghis Khan?"
8. "Tell me about the Persian Empire according to the lectures."
9. "What does Jiang think about Napoleon?"
10. "How does Jiang explain the Bronze Age collapse?"

### Médias (requer combinar informação de múltiplos chunks)
11. "What are the main causes of the fall of Rome according to Jiang?"
12. "How does Jiang explain the rise of the Roman Republic?"
13. "What factors does Jiang identify in the success of the Mongol Empire?"
14. "What is Jiang's analysis of why empires fall?"
15. "How does Jiang connect religion to the fall of civilizations?"
16. "What role does Jiang assign to economics in historical change?"
17. "What does Jiang say about the relationship between military power and empire?"
18. "How does Jiang explain the transition from Republic to Empire in Rome?"

### Difíceis (requer raciocínio sobre claims e relações)
19. "What causal chain does Jiang propose for the fall of the Western Roman Empire?"
20. "According to Jiang, what patterns repeat across different empire collapses?"
21. "What is Jiang's theory about the relationship between internal decay and external invasion?"
22. "How does Jiang connect the Greek city-states to the rise of Macedon?"

---

## Comparative Queries (Claim Search + Cypher Templates)

### Comparação de pessoas
23. "How does Jiang compare Alexander the Great and Julius Caesar as military leaders?"
24. "What are the differences between Caesar and Augustus according to Jiang?"
25. "How does Jiang compare Napoleon and Alexander?"
26. "What similarities does Jiang see between Hannibal and Alexander?"
27. "How does Jiang contrast Sparta and Athens?"

### Comparação de impérios/organizações
28. "How does Jiang compare the Roman Empire and the Persian Empire?"
29. "What are the differences between the Roman Republic and Athenian Democracy?"
30. "How does Jiang compare the Mongol Empire to the Roman Empire?"
31. "What parallels does Jiang draw between ancient empires and modern nations?"
32. "How does Jiang compare the fall of Rome to the potential decline of the US?"

### Comparação de conceitos/abordagens
33. "How does Jiang compare democracy and autocracy as governing systems?"
34. "What does Jiang say about land warfare vs. naval warfare?"
35. "How does Jiang contrast Eastern and Western approaches to governance?"

---

## Temporal Queries (Chronological Claim Aggregation)

### Evolução de previsões
36. "How has Jiang's prediction about the United States evolved over the episodes?"
37. "Has Jiang changed his mind about China's future?"
38. "Track Jiang's predictions about modern geopolitics across all episodes."
39. "What predictions has Jiang made that were later confirmed or refuted?"
40. "How has Jiang's analysis of Russia evolved?"

### Evolução de interpretações
41. "Has Jiang's view on the causes of WWI changed over time?"
42. "How has Jiang's interpretation of the Roman Republic evolved?"
43. "Track how Jiang discusses democracy across different episodes."
44. "Has Jiang revised any of his historical interpretations?"

### Temporal com data específica
45. "What did Jiang predict in his earliest episodes?"
46. "What topics dominated the most recent episodes?"
47. "What claims did Jiang make in episodes 1-10 that he revisited later?"

---

## Thematic Queries (Community Summaries)

### Temas gerais
48. "What are the main themes across all of Jiang's lectures?"
49. "What are the dominant topics in the Predictive History channel?"
50. "Give me an overview of all subjects Jiang covers."
51. "What historical periods does Jiang focus on most?"
52. "What are the recurring patterns Jiang identifies across history?"

### Temas específicos
53. "What cluster of topics relates to the Mediterranean world?"
54. "What themes connect Jiang's discussions of ancient warfare?"
55. "What group of ideas does Jiang associate with civilizational decline?"
56. "What topics cluster around Jiang's geopolitical predictions?"

---

## Analytical / Ad-Hoc Queries (Text-to-Cypher)

### Contagens e rankings
57. "How many episodes discuss the Roman Empire?"
58. "Which entities are mentioned most frequently across all lectures?"
59. "What are the top 10 most discussed historical figures?"
60. "How many predictions has Jiang made in total?"
61. "What percentage of Jiang's predictions have been verified?"
62. "Which episodes have the most extracted claims?"
63. "What topics have the most causal analysis claims?"

### Estruturais (exploram o grafo)
64. "What events does Jiang connect causally to World War I?"
65. "Which historical figures are connected to the most events?"
66. "What organizations controlled the most territories?"
67. "Show me all causal chains longer than 3 events."
68. "Which claims contradict each other?"
69. "What entities appear in the most communities?"
70. "Which topics span the most episodes?"

### Complexas (multi-hop)
71. "What persons participated in events that led to other events involving different persons?"
72. "Find claims about entities that are in the same community."
73. "What predictions reference events that are causally connected?"

---

## Edge Cases e Queries Difíceis

### Queries que o sistema deveria admitir não saber
74. "What does Jiang think about quantum physics?" (fora do domínio)
75. "What did Jiang say in episode 500?" (episódio não existe)
76. "Compare Jiang's views with those of another historian." (não temos outro historiador)

### Queries ambíguas
77. "Tell me about Rome." (República? Império? Cidade? Religião?)
78. "What about the war?" (qual guerra?)
79. "What did he predict?" (sobre o quê?)

### Queries que testam coreference resolution
80. "What did Jiang say about the Macedonian king's conquests?" (= Alexander)
81. "Tell me about the philosopher who taught Alexander." (= Aristotle)
82. "What happened to the empire after the conqueror died?" (context-dependent)

### Queries que testam entity resolution
83. "What did Jiang say about Philip II?" (Macedon or Spain?)
84. "Tell me about the Republic." (Roman? French? Athenian?)
85. "What does Jiang think about the Empire?" (which one?)

---

## Queries para Validação de Qualidade

### Provenance (o sistema deve citar fontes)
86. "In which episode did Jiang first discuss the fall of Rome?"
87. "When did Jiang make his prediction about the US?"
88. "What episode contains Jiang's analysis of Alexander's strategy?"

### Completeness (o sistema deve agregar de múltiplas fontes)
89. "Give me EVERYTHING Jiang has said about Julius Caesar."
90. "Compile all predictions Jiang has made."
91. "List all causal explanations Jiang has given for historical events."

### Accuracy (respostas devem ser fiéis ao que Jiang disse)
92. "Does Jiang think the Roman Empire fell because of Christianity?" (verify against actual claims)
93. "Did Jiang predict a specific date for any geopolitical event?" (check target_date fields)
94. "What is Jiang's stance on American democracy?" (check stance field)

---

## Queries de Stress Test

### Volume
95. "List all entities in the knowledge graph." (should cap results)
96. "Show me every claim ever made." (should cap results)
97. "Return all relationships." (should cap results)

### Malformed
98. "" (empty query)
99. "asdfghjkl" (nonsense)
100. "DROP DATABASE" (injection attempt — should be safe)

---

## Notas de Uso

- Rodar estas queries APÓS ingerir pelo menos 5 episódios
- Documentar quais queries funcionam bem e quais falham
- Queries que falham indicam onde o pipeline precisa de ajuste
- Usar os resultados para iterar nos prompts de extração
- Registrar resultados no TCC.md como parte da avaliação
