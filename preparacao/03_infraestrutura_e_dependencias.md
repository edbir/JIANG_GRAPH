# Preparação — Infraestrutura e Dependências

## Objetivo
Listar tudo que precisamos ter instalado, configurado, e com acesso antes de começar a implementar.

---

## 1. Banco de Dados de Grafos — Neo4j

**O que é:** O banco onde o knowledge graph vai viver. Armazena entidades, relações, embeddings, e comunidades.

**Opções:**
- **Neo4j Desktop** (gratuito, local) — recomendado para desenvolvimento
- **Neo4j via Docker** — mesma coisa que local, mas roda dentro de um container

**Neo4j Desktop vs. Docker — qual a diferença:**
- **Neo4j Desktop** é um aplicativo com interface gráfica que gerencia instâncias Neo4j para você. Você clica em "New Database", ele cria tudo. Mais fácil para quem não usa Docker no dia a dia. Inclui Neo4j Browser embutido.
- **Neo4j Docker** é rodar o Neo4j como um container. Vantagem: reprodutibilidade (qualquer pessoa pode rodar `docker-compose up` e ter o mesmo ambiente). Desvantagem: precisa ter Docker instalado e saber o básico de containers.
- **Para o MVP:** qualquer um dos dois funciona. Se você já usa Docker, vá de Docker. Se não, Neo4j Desktop é mais direto.

**Requisitos:**
- Neo4j 5.x+ (necessário para vector indexes nativos)
- Plugin GDS (Graph Data Science) — necessário para Leiden community detection
  - ATENÇÃO: GDS não está disponível no Aura Free. Para usar Leiden, precisa ser Desktop ou self-hosted.

**Instalação sugerida:**
- Neo4j Desktop: https://neo4j.com/download/
- Ou via Docker: `docker run -p 7474:7474 -p 7687:7687 neo4j:5`

**O que configurar:**
- Criar um database para o projeto
- Habilitar o plugin GDS
- Anotar: bolt URI (geralmente `bolt://localhost:7687`), usuário e senha

---

## 2. Chaves de API — LLMs e Embeddings

### 2.1 OpenAI API (LLM + Embeddings)
**O que é:** A API da OpenAI será nosso único provider de LLM. Isso simplifica o projeto e elimina variabilidade na pesquisa.

**Obter em:** https://platform.openai.com/api-keys

**Modelos que vamos usar:**
- `gpt-4o-mini` — Passes 1-2 de extração (entidades, relações), entity resolution, query classification
- `gpt-4o` — Pass 3 (claims/opiniões, nuance de stance) e Pass 4 (coreference resolution)
- `text-embedding-3-small` (1536 dimensões) — Todos os embeddings:
  - Embeddings de utterances (chunks)
  - Embeddings de claims
  - Embeddings de entidades (nome + descrição)
  - Embeddings de community summaries

**Custo estimado embeddings:** ~$0.02 por 1M tokens — muito barato

**Decisão:** Usar OpenAI diretamente, sem OpenRouter, sem Claude. Um provider, menos variáveis.

### 2.2 Groq API (para transcrição — prova de conceito)
**O que é:** Groq oferece um free tier que usa Whisper para transcrição de áudio. Vamos usar como prova de conceito para os primeiros episódios.

**Obter em:** https://console.groq.com/

**Notas:**
- O free tier tem limites de uso (requests/minuto e tamanho de arquivo)
- Para o MVP de 5 episódios é suficiente
- Não é uma dependência permanente do projeto — é uma ferramenta para obter as transcrições iniciais
- Se escalarmos para 134 episódios, podemos reavaliar (legendas do YouTube, Whisper local, ou API paga)

**Decisão:** Groq para MVP. Transcrição não é o foco da pesquisa — é um pré-requisito operacional.

---

## 3. Linguagem e Ambiente de Desenvolvimento

### 3.1 Python 3.11+
**Por que:** Todo o ecossistema de LLMs, Neo4j, e MCP tem suporte primário em Python.

**Gerenciamento de dependências:** 
- `uv` (recomendado, rápido) ou `poetry` ou `pip + venv`

### 3.2 Bibliotecas Python principais
```
# Core
neo4j                  # Driver oficial Neo4j
pydantic               # Schemas de extração (structured outputs)

# LLM e Embeddings
openai                 # SDK OpenAI (completions + embeddings)

# Entity Resolution
rapidfuzz              # Fuzzy string matching
numpy                  # Operações com embeddings

# MCP Server
mcp                    # FastMCP para criar o servidor MCP

# Utilitários
tiktoken               # Contagem de tokens (para chunking)
python-dotenv          # Gerenciamento de variáveis de ambiente
yt-dlp                 # Download de áudio do YouTube
```

---

## 4. Ferramentas de Desenvolvimento

### 4.1 IDE
- Kiro (já estamos usando)

### 4.2 Git
- Controle de versão para o código e para o TCC.md

### 4.3 Neo4j Browser
- Interface web para visualizar o grafo e testar queries Cypher
- Vem embutido no Neo4j Desktop (http://localhost:7474)

### 4.4 (Opcional) Jupyter Notebook
- Útil para experimentação interativa durante desenvolvimento dos prompts de extração
- Permite ver resultados de extração chunk por chunk

---

## 5. Dados de Entrada — Transcrições

**Fonte:** Episódios do canal Predictive History no YouTube

**Formato esperado após transcrição:**
- Texto com timestamps (início/fim de cada segmento)
- Idealmente com marcação de speaker (mesmo que seja só "Jiang" na maioria)

**Pipeline de obtenção:**
1. Baixar áudio dos vídeos (yt-dlp)
2. Transcrever (Whisper API, Gemini, ou legendas existentes)
3. Limpar e normalizar o texto
4. Armazenar como arquivos JSON ou texto na pasta `data/transcripts/`

**Ferramenta para download:**
- `yt-dlp` — ferramenta de linha de comando para baixar áudio/vídeo do YouTube

---

## 6. Resumo de Custos Estimados

| Item | Custo |
|------|-------|
| Neo4j Desktop/Community | Gratuito |
| OpenAI API (extração + embeddings, 5 ep MVP) | ~$1–3 |
| OpenAI API (extração + embeddings, 134 ep full) | ~$8–27 |
| Transcrição via Groq (MVP, 5 ep) | Gratuito (free tier) |
| Transcrição (134 ep, se necessário pagar) | ~$10–30 |
| Neo4j GDS | Gratuito (Community/Desktop) |
| **Total MVP (5 episódios)** | **~$1–3** |
| **Total full (134 episódios)** | **~$18–57** |

---

## 7. Variáveis de Ambiente Necessárias (.env)

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<sua_senha>

# LLM + Embeddings (OpenAI)
OPENAI_API_KEY=<sua_chave>

# Transcrição (Groq — apenas para MVP)
GROQ_API_KEY=<sua_chave>
```

---

## 8. Decisões Tomadas

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Provider de LLM | OpenAI direto | Simplicidade, elimina variabilidade na pesquisa |
| Embeddings | OpenAI `text-embedding-3-small` | Custo-efetivo, boa qualidade, mesmo provider |
| Transcrição (MVP) | Groq free tier (Whisper) | Gratuito, suficiente para 5 episódios |
| Escopo MVP | 5 episódios | Validar pipeline antes de escalar |
| Neo4j | Docker | Reprodutibilidade, familiaridade do aluno com Docker |

Todas as decisões de infraestrutura estão fechadas. Podemos começar a implementar.
