# Chunker — Documentação Técnica

## O que é

`src/preprocessing/chunker.py` divide uma transcrição limpa em chunks prontos para extração. Usa uma estratégia de dois passes (Two-Pass Chunking).

## Posição no pipeline

```
text_cleaner.py → [chunker.py] → list[Chunk] → extractor.py → ...
```

## Os dois passes

### Level 1 — Segment Detection (usa LLM)

**Pergunta:** "Onde o professor muda de tópico?"

- Divide o texto em sentenças
- Agrupa sentenças em janelas de ~20
- Para cada janela, pergunta ao GPT-4o-mini onde há mudanças de tópico
- Constrói Segments (blocos temáticos de tamanho variável)

**Output:** `list[Segment]` — cada um com sentences, topic, e offsets

### Level 2 — Token-Bounded Chunking (determinístico, sem LLM)

**Pergunta:** "Como dividir cada segmento em pedaços de ~1500 tokens?"

- Acumula sentenças até ~1500 tokens
- Nunca corta no meio de uma sentença
- Overlap de 3 sentenças entre chunks consecutivos
- Chunks muito pequenos (<200 tokens) são absorvidos pelo anterior

**Output:** `list[Chunk]` — prontos para extração

## Estruturas de dados

```python
@dataclass
class Segment:
    segment_index: int      # Posição no episódio
    sentences: list[str]    # Sentenças do segmento
    topic: str              # Descrição do tópico (do LLM)
    start_char: int         # Offset no texto original
    end_char: int

@dataclass
class Chunk:
    chunk_id: str           # "ep001-seg02-chunk03"
    text: str               # O texto
    episode_id: str         # Qual episódio
    segment_index: int      # Qual segmento
    chunk_index: int        # Posição no segmento
    global_index: int       # Posição no episódio inteiro
    token_count: int        # Tokens estimados
    start_char: int         # Offset início
    end_char: int           # Offset fim
    overlap_with_previous: bool
```

## Correspondência com o grafo Neo4j

- `Chunk` → nó `Utterance` (utteranceId = chunk_id)
- `Segment` → nó `Segment` (segmentId, topic)
- A hierarquia Episode → Segment → Utterance é preservada pela metadata

## Como usar

```python
from src.preprocessing.chunker import chunk_transcript

# Com LLM (produção)
chunks = await chunk_transcript(text, "ep001", llm_client=client)

# Sem LLM (teste — segmentos uniformes)
chunks = await chunk_transcript(text, "ep001", llm_client=None)
```

## Funções-chave

| Função | O que faz | Precisa de API? |
|--------|-----------|-----------------|
| `split_into_sentences()` | Divide texto em sentenças | Não |
| `prepare_segmentation_windows()` | Agrupa sentenças em janelas | Não |
| `segment_with_llm()` | Chama LLM para detectar tópicos | Sim |
| `segment_with_llm_mock()` | Segmentos uniformes para teste | Não |
| `build_segments_from_boundaries()` | Constrói Segments dos índices | Não |
| `chunk_segment()` | Level 2: split com overlap | Não |
| `chunk_transcript()` | Entry point completo | Parcial |

## Parâmetros (de config.py)

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| max_chunk_tokens | 1500 | Máximo tokens por chunk |
| min_chunk_tokens | 200 | Mínimo para chunk independente |
| overlap_sentences | 3 | Overlap entre chunks |
| segmentation_window_size | 20 | Sentenças por janela LLM |
| segmentation_window_overlap | 3 | Overlap entre janelas |
