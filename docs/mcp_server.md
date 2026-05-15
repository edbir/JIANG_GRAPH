# MCP Server — Documentação Técnica

## O que é

`src/mcp/server.py` expõe o knowledge graph para LLMs externos via Model Context Protocol. É a interface final do sistema — o que o usuário (via Claude, GPT, etc.) interage.

## Os 6 Tools

| Tool | Propósito | Quando usar |
|------|-----------|-------------|
| `search_lectures` | Busca semântica geral | "What did Jiang say about X?" |
| `find_claims` | Busca claims com filtros | "What predictions did Jiang make?" |
| `compare_entities` | Compara entidades lado a lado | "How do A and B differ on X?" |
| `topic_timeline` | Evolução cronológica de claims | "How has the view on X evolved?" |
| `run_cypher` | NL → Cypher para queries ad-hoc | "How many episodes mention X?" |
| `get_schema` | Retorna o schema do grafo | "What can I query?" |

## Design

- **Domain tools** (search, find_claims, compare, timeline): otimizados, rápidos, fáceis de usar corretamente
- **Generic tools** (run_cypher, get_schema): flexibilidade para queries não antecipadas
- **Caps em resultados**: cada tool limita output para não estourar context do client
- **Structured data**: tools retornam JSON; o LLM client faz a geração em linguagem natural

## Como rodar

```bash
python -m src.mcp.server
```

Roda em stdio (stdin/stdout). O client MCP (Claude Desktop, Kiro, etc.) conecta via processo.

## Configuração no client

Para usar com Kiro/Claude Desktop, adicione ao `mcp.json`:

```json
{
  "mcpServers": {
    "jiang-graphrag": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/JIANG_GRAPH"
    }
  }
}
```

## Dependências em runtime

- Neo4j rodando (docker compose up -d)
- OPENAI_API_KEY no .env
- NEO4J_PASSWORD no .env
- Grafo populado (pelo menos 1 episódio ingerido)

## Fluxo interno de cada tool

```
Client LLM chama tool (ex: search_lectures("fall of Rome"))
    → _ensure_clients() (inicializa OpenAI + Neo4j se necessário)
    → Chama a função de retrieval correspondente
    → Formata resultado como JSON
    → Retorna ao client LLM
    → Client LLM gera resposta em linguagem natural para o usuário
```

## Separação de responsabilidades

O MCP server NÃO faz geração de texto. Ele retorna DADOS estruturados. O LLM client-side (Claude, GPT) é quem transforma esses dados em uma resposta legível para o usuário. Isso permite que diferentes clients usem diferentes system prompts e estilos de resposta.
