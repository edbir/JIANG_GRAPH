# Config — Documentação Técnica

## O que é

`src/config.py` centraliza TODAS as configurações do projeto em um único lugar. Usa Pydantic Settings para ler automaticamente do `.env`.

## Como funciona

```python
from src.config import settings

# Usar qualquer configuração:
settings.openai_api_key
settings.max_chunk_tokens      # 1500
settings.model_cheap           # "gpt-4o-mini"
settings.fuzzy_match_threshold # 85
```

## Prioridade de leitura

1. Variável de ambiente real (ex: `export OPENAI_API_KEY=sk-...`) — maior prioridade
2. Arquivo `.env` na raiz do projeto — fallback
3. Valor default definido no código — menor prioridade

## Categorias de configuração

### API Keys
| Campo | Default | Descrição |
|-------|---------|-----------|
| openai_api_key | "" | Chave OpenAI (LLM + embeddings) |
| groq_api_key | "" | Chave Groq (transcrição MVP) |

### Neo4j
| Campo | Default | Descrição |
|-------|---------|-----------|
| neo4j_uri | bolt://localhost:7687 | URI de conexão |
| neo4j_user | neo4j | Usuário |
| neo4j_password | "" | Senha |

### Modelos LLM
| Campo | Default | Descrição |
|-------|---------|-----------|
| model_cheap | gpt-4o-mini | Alto volume (Pass 1, 2, 5) |
| model_strong | gpt-4o | Nuance (Pass 3, 4) |
| model_segmentation | gpt-4o-mini | Chunker Level 1 |
| embedding_model | text-embedding-3-small | Embeddings |
| embedding_dimensions | 1536 | Dimensões do vetor |

### Chunking
| Campo | Default | Descrição |
|-------|---------|-----------|
| max_chunk_tokens | 1500 | Máximo tokens/chunk |
| min_chunk_tokens | 200 | Mínimo tokens/chunk |
| overlap_sentences | 3 | Overlap entre chunks |
| segmentation_window_size | 20 | Sentenças por janela LLM |
| segmentation_window_overlap | 3 | Overlap entre janelas |

### Extração
| Campo | Default | Descrição |
|-------|---------|-----------|
| extraction_confidence_threshold | 0.7 | Mínimo para ingerir |
| coreference_window_chunks | 4 | Chunks na janela do Pass 4 |

### Entity Resolution
| Campo | Default | Descrição |
|-------|---------|-----------|
| fuzzy_match_threshold | 85 | RapidFuzz score mínimo |
| embedding_similarity_threshold | 0.85 | Cosine mínimo |

## Dependência

Requer `pydantic-settings` (`pip install pydantic-settings`).
