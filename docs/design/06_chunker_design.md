# Design do Chunker — Two-Pass Chunking Strategy

## O Problema

Temos transcrições de aulas de ~1 hora do Prof. Jiang. Cada transcrição tem ~15,000-20,000 tokens. Precisamos dividir esse texto em pedaços (chunks) que:

1. Caibam na context window do LLM para extração (~1500 tokens por chunk)
2. Preservem a coerência semântica — não cortar no meio de uma ideia
3. Mantenham contexto suficiente para extração de entidades e relações
4. Permitam overlap para capturar assertions que cruzam fronteiras de chunk

## Por que Chunking Fixo Falha

O chunking de tamanho fixo (ex: "corte a cada 1500 tokens") é o padrão em RAG tradicional. Ele falha para nosso caso porque:

- **Corta ideias no meio:** O professor pode estar explicando uma cadeia causal complexa ("A causou B, que levou a C, resultando em D") e o corte fixo pode separar a causa do efeito.
- **Perde contexto de referência:** Se o professor diz "Alexander" no início de uma explicação e usa "he" por 500 tokens, um corte no meio perde a referência.
- **Ignora estrutura do discurso:** Aulas têm estrutura natural — o professor muda de tópico, faz transições, abre e fecha argumentos. Chunking fixo ignora tudo isso.

## A Solução: Two-Pass Chunking

Nossa estratégia usa dois passes:

### Level 1 — Segment Detection via LLM (onde as ideias começam e terminam)

O primeiro passo usa um LLM para identificar **fronteiras naturais de tópico** no texto. Passamos blocos de sentenças ao modelo e perguntamos onde os tópicos mudam. Isso produz "segmentos" de tamanho variável (podem ter 500 tokens ou 5000 tokens).

**Por que LLM e não heurísticas:**
- Marcadores discursivos (regex) são frágeis — o professor pode mudar de assunto sem usar nenhum marcador explícito, ou pode usar um marcador sem de fato mudar de tópico ("anyway" nem sempre indica transição).
- Embedding similarity (cosine drop) é melhor que regex mas ainda é superficial — detecta mudança de vocabulário, não necessariamente mudança de argumento.
- O LLM entende semântica: ele sabe que "a queda de Roma" e "o colapso do Império Romano" são o mesmo tópico, e que uma transição de "economia romana" para "guerras púnicas" é uma mudança real de assunto.

**Como funciona:**
1. Dividimos o texto em sentenças (splitting determinístico por pontuação)
2. Agrupamos sentenças em janelas de ~15-20 sentenças
3. Para cada janela, chamamos o LLM pedindo os índices onde há mudança de tópico
4. Construímos os segmentos a partir dos índices retornados

**Custo:** Uma chamada LLM (GPT-4o-mini, barato) por janela de ~15-20 sentenças. Para um episódio de ~500 sentenças, são ~25-33 chamadas. Com prompt caching, o custo é negligível.

**Prompt do LLM para segmentação:**
- System: "You are analyzing a history lecture transcript. Identify where the speaker changes topic."
- User: As sentenças numeradas + instrução para retornar os índices de boundary
- Output: Lista de índices onde novos tópicos começam + breve descrição do tópico

### Level 2 — Token-Bounded Chunking (chunks de ~1500 tokens com overlap)

Dentro de cada segmento do Level 1, dividimos em chunks de no máximo ~1500 tokens, respeitando:
- **Fronteiras de sentença:** Nunca cortar no meio de uma frase
- **Overlap:** 2-3 sentenças de overlap entre chunks consecutivos para capturar assertions cross-boundary
- **Mínimo:** Chunks menores que ~200 tokens são absorvidos pelo chunk anterior (evitar micro-chunks sem contexto)

Se um segmento Level 1 já tem ≤1500 tokens, ele vira um único chunk Level 2 sem subdivisão.

Este passo é **determinístico** — não usa LLM. É pura lógica de acumulação de sentenças até o limite de tokens.

## Estrutura de Dados do Output

Cada chunk produzido carrega metadata:

```python
{
    "chunk_id": "ep001-seg02-chunk03",    # Identificador único
    "text": "...",                          # O texto do chunk
    "episode_id": "ep001",                 # A qual episódio pertence
    "segment_index": 2,                    # A qual segmento Level 1 pertence
    "segment_topic": "The fall of...",     # Tópico do segmento (do LLM)
    "chunk_index": 3,                      # Posição dentro do segmento
    "global_index": 15,                    # Posição global no episódio
    "token_count": 1423,                   # Contagem de tokens
    "start_char": 12450,                   # Offset no texto original (para provenance)
    "end_char": 14320,                     # Offset final
    "overlap_with_previous": True,         # Se tem overlap com o chunk anterior
}
```

## Parâmetros Configuráveis

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `max_chunk_tokens` | 1500 | Tamanho máximo de um chunk Level 2 |
| `min_chunk_tokens` | 200 | Mínimo para um chunk ser independente |
| `overlap_sentences` | 3 | Sentenças de overlap entre chunks consecutivos |
| `segmentation_window_size` | 20 | Sentenças por janela enviada ao LLM no Level 1 |
| `segmentation_window_overlap` | 3 | Overlap de sentenças entre janelas do Level 1 |

## O Papel do Overlap

O overlap de 2-3 sentenças entre chunks consecutivos é crucial. A pesquisa (Rook, ISWC 2025) demonstra que em dados de diálogo, as distâncias subject-object span são maiores que em documentos — ou seja, o sujeito de uma relação pode estar em uma frase e o objeto em outra, separados por várias sentenças.

O overlap garante que assertions que cruzam a fronteira de um chunk sejam capturadas em pelo menos um dos dois chunks adjacentes. Sem overlap, perderíamos relações e claims que estão "na fronteira".

## Relação com o Schema do Grafo

Os chunks Level 2 correspondem aos nós **Utterance** no grafo Neo4j:
- `chunk_id` → `utteranceId`
- `text` → `text`
- `token_count` → `tokenCount`
- `embedding` → gerado depois, armazenado como `embedding`

Os segmentos Level 1 correspondem aos nós **Segment**:
- `segment_index` → `segmentId`
- `segment_topic` → `topic` (vem do LLM na segmentação)
- `start_time` / `end_time` → se a transcrição tiver timestamps

A hierarquia Episode → Segment → Utterance (chunk) é preservada pela metadata.

## Separação de Responsabilidades no Código

O chunker é estruturado em camadas independentes:

1. **Sentence splitting** — determinístico, sem dependência externa
2. **LLM segmentation call** — isolada em uma função que pode ser mockada/substituída
3. **Segment construction** — recebe os índices do LLM e constrói os objetos Segment
4. **Token-bounded chunking** — determinístico, opera sobre segmentos já construídos
5. **Pipeline completo** — orquestra tudo

Isso permite:
- Testar o Level 2 (chunking) sem API (passando segmentos manualmente)
- Substituir o LLM de segmentação por outro método no futuro
- Debugar cada camada independentemente

## Notas para o TCC

### O que documentar no artigo:
1. **O problema do chunking fixo** — com exemplos concretos de informação perdida
2. **A estratégia two-pass** — Level 1 (segmentação semântica via LLM) + Level 2 (token-bounded)
3. **Por que LLM para segmentação** — superioridade sobre heurísticas e embeddings para detecção de mudança de tópico em discurso falado
4. **O papel do overlap** — com referência ao Rook (ISWC 2025) sobre span distances em diálogo
5. **Trade-offs:** custo adicional do LLM no Level 1 vs. qualidade dramaticamente superior dos segmentos
6. **A correspondência com o schema do grafo** — chunks = Utterance nodes, segmentos = Segment nodes
