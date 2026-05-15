# Design do Upsert — Inserção Idempotente no Neo4j

## O Problema

Após extração e entity resolution, temos entidades, relações, e claims prontos para ir ao grafo. Precisamos inseri-los de forma que:

1. **Não crie duplicatas** — se rodarmos o pipeline duas vezes no mesmo episódio, o grafo não deve ter dados duplicados
2. **Enriqueça dados existentes** — se uma entidade já existe mas a nova extração tem uma descrição melhor, atualize
3. **Preserve provenance** — cada claim sabe de qual utterance veio, cada utterance sabe de qual segmento/episódio
4. **Seja atômica por episódio** — ou todo o episódio é inserido, ou nada (evitar estados parciais)

## A Solução: MERGE Patterns

Neo4j tem o comando `MERGE` que funciona como "find or create":
- Se o nó/relação já existe (pelo ID), ele é encontrado (não cria duplicata)
- Se não existe, é criado
- `ON CREATE SET` define propriedades apenas na criação
- `ON MATCH SET` define propriedades quando o nó já existia (atualização)

## A Regra de Ouro do MERGE

**NUNCA fazer MERGE de um pattern inteiro com variáveis não-bound.**

```cypher
-- ❌ ERRADO: pode criar nós duplicados
MERGE (a:Person {name: "Caesar"})-[:ROLE_IN]->(b:Organization {name: "Rome"})

-- ✅ CORRETO: bind nós primeiro, depois MERGE a relação
MATCH (a:Person {personId: "person-caesar"})
MATCH (b:Organization {orgId: "org-rome"})
MERGE (a)-[:ROLE_IN]->(b)
```

No padrão errado, se a relação não existe, Neo4j cria NOVOS nós a e b (mesmo que já existam). No padrão correto, encontramos os nós existentes e só criamos a relação.

## Ordem de Inserção

A ordem importa porque relações referenciam nós que devem existir:

1. **Episode** — o container de tudo
2. **Segments** — pertencem ao episódio
3. **Utterances (chunks)** — pertencem aos segmentos
4. **Entities** (Person, Organization, Topic, Event) — mencionadas nos chunks
5. **Claims** — extraídas dos chunks, referenciam entidades
6. **Relationships** entre entidades — conectam entidades já existentes
7. **Claim relationships** (SUPPORTS, CONTRADICTS) — conectam claims entre si
8. **Community memberships** — após community detection

## Estratégia de IDs

Cada nó precisa de um ID único e determinístico (para que MERGE funcione):

| Nó | ID Format | Exemplo |
|----|-----------|---------|
| Episode | `ep{number:03d}` | `ep001` |
| Segment | `{episodeId}-seg{index:02d}` | `ep001-seg03` |
| Utterance | `{episodeId}-seg{segIdx:02d}-chunk{chunkIdx:02d}` | `ep001-seg03-chunk02` |
| Person | `person-{normalized_name}` | `person-alexander-the-great` |
| Organization | `org-{normalized_name}` | `org-roman-empire` |
| Topic | `topic-{normalized_name}` | `topic-democracy` |
| Event | `event-{normalized_name}` | `event-battle-of-thermopylae` |
| Claim | `claim-{hash(text)[:12]}` | `claim-a3f2b9c1d4e5` |
| Community | `community-{level}-{id}` | `community-0-42` |

IDs de entidades usam o nome canônico normalizado (após entity resolution). IDs de claims usam hash do texto (determinístico, único).

## Enriquecimento de Descrição

Quando uma entidade já existe e é encontrada novamente com uma descrição diferente, mantemos a MAIS LONGA (geralmente mais informativa):

```cypher
MERGE (e:Person {personId: $id})
ON CREATE SET e.name = $name, e.description = $desc, e.sourceCount = 1
ON MATCH SET
    e.description = CASE
        WHEN size(e.description) < size($desc) THEN $desc
        ELSE e.description
    END,
    e.sourceCount = e.sourceCount + 1
```

O `sourceCount` rastreia em quantos chunks a entidade foi mencionada — entidades com alto sourceCount são mais centrais no grafo.

## Embeddings

Após inserir todos os nós, geramos embeddings e os armazenamos:

- Utterance: embedding do texto completo do chunk
- Claim: embedding de "claim text + entities involved"
- Entity: embedding de "name: description (aliases)"
- Community: embedding do summary (gerado depois)

Isso é feito em batch após o upsert, não durante.

## Notas para o TCC

### O que documentar no artigo:
1. **Idempotência via MERGE** — por que é essencial para pipelines de dados
2. **A regra de ouro** — bind nodes first, then MERGE relationships
3. **Enriquecimento progressivo** — descrições melhoram conforme mais episódios são processados
4. **Ordem de inserção** — por que a hierarquia importa
5. **Deterministic IDs** — reprodutibilidade e deduplicação
