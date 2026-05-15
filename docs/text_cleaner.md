# Text Cleaner — Documentação Técnica

## O que é

`src/preprocessing/text_cleaner.py` é o primeiro passo do pipeline. Recebe uma transcrição bruta (do Groq/Whisper) e produz texto limpo e consistente.

## Posição no pipeline

```
Raw transcript → [text_cleaner.py] → Clean text → chunker.py → ...
```

## Por que existe

Transcrições brutas contêm ruído que:
- Desperdiça tokens (fillers como "uh", "you know" custam dinheiro na API)
- Confunde a extração (LLM vê "[inaudible] conquered [unclear]")
- Quebra o sentence splitting (falsos limites de sentença)

## O que faz (5 etapas, em ordem)

| Etapa | Função | Exemplo |
|-------|--------|---------|
| 1. Unicode | `normalize_unicode()` | "smart quotes" → standard quotes |
| 2. Artifacts | `remove_transcription_artifacts()` | "[inaudible]", "[00:15:32]" → removido |
| 3. Fillers | `remove_filler_words()` | "uh", "uhm", ", you know," → removido |
| 4. Stutters | `remove_stutters()` | "the the Roman" → "the Roman" |
| 5. Whitespace | `normalize_whitespace()` | espaços duplos, pontuação solta → limpo |

## O que NÃO faz

- Não altera conteúdo semântico
- Não corrige erros do transcritor
- Não traduz
- Não remove conteúdo "irrelevante" (isso é decisão do extractor)

## Como usar

```python
from src.preprocessing.text_cleaner import clean_transcript

raw = "[00:01:15] So, uh, the the Roman Empire, you know, it fell because..."
clean = clean_transcript(raw)
# → "So, the Roman Empire, it fell because..."
```

## Decisões de design

- **Fillers contextuais:** "like", "you know", "basically" só são removidos quando aparecem em padrões de filler (entre vírgulas, no início de frase). "Something like a republic" NÃO é removido.
- **Ordem importa:** Unicode primeiro (para regex funcionar), whitespace por último (para limpar gaps).
- **Zero dependências externas:** Só usa `re` e `unicodedata` da stdlib.
