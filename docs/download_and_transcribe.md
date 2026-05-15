# Download and Transcribe

## O que é

`scripts/download_and_transcribe.py` é o **primeiro passo** do pipeline. Baixa áudio de vídeos do YouTube (via yt-dlp) e transcreve usando a API Whisper da Groq.

## Posição no pipeline

```
YouTube URL → [download_and_transcribe.py] → transcript.txt → ingest_episode.py → ...
```

Este script produz o arquivo de texto que `ingest_episode.py` consome.

## Pré-requisitos

### Software
- **yt-dlp**: `pip install yt-dlp` (já está no requirements.txt)
- **ffmpeg**: necessário para conversão de áudio e split de arquivos longos
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

### API
- **GROQ_API_KEY** no `.env` — obtém grátis em https://console.groq.com

## Uso

### Vídeo único
```bash
python -m scripts.download_and_transcribe \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --episode-id ep001
```

### Manter o áudio (não deleta após transcrição)
```bash
python -m scripts.download_and_transcribe \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --episode-id ep001 \
    --keep-audio
```

### Batch (múltiplos episódios)
```bash
python -m scripts.download_and_transcribe \
    --batch episodes.txt
```

**Formato do arquivo batch** (`episodes.txt`):
```
# Linhas com # são comentários
ep001 https://www.youtube.com/watch?v=ABC123 The Fall of Rome
ep002 https://www.youtube.com/watch?v=DEF456 Alexander's Conquests
ep003 https://www.youtube.com/watch?v=GHI789 The Bronze Age Collapse
```

## O que faz (passo a passo)

```
1. Download do áudio (yt-dlp)
   - Extrai apenas áudio (sem vídeo)
   - Converte para mp3 a 64kbps (suficiente para fala)
   - ~29 MB por hora de aula

2. Split do áudio (ffmpeg, se necessário)
   - Groq aceita no máximo 25 MB / ~25 min por request
   - Aulas de ~1 hora são divididas em segmentos de 20 min
   - Split é sem re-encoding (rápido)

3. Transcrição (Groq Whisper large-v3)
   - Cada segmento é enviado separadamente
   - Resultados concatenados na ordem correta
   - Idioma: inglês (hardcoded, já que Jiang fala em inglês)

4. Salva como texto plano
   - UTF-8, em data/transcripts/{episode_id}.txt
   - Pronto para ingest_episode.py consumir

5. Cleanup
   - Remove áudio (a menos que --keep-audio)
   - Remove segmentos temporários
```

## Custo

| Item | Custo |
|------|-------|
| yt-dlp download | Grátis |
| Groq Whisper (free tier) | Grátis (14,400 seg/dia ≈ 4 horas) |
| Groq Whisper (pago) | $0.111 / hora de áudio |

**Free tier**: 4 episódios/dia → 134 episódios em ~34 dias.

## Limites do Groq Free Tier

- 14,400 audio-seconds por dia (~4 horas)
- 25 MB por arquivo
- Requests por minuto: 20

O script lida com o limite de 25 MB automaticamente (split em segmentos). Os rate limits diários precisam ser gerenciados manualmente — processe 3-4 episódios por dia.

## Output

```
data/
├── transcripts/
│   ├── ep001.txt    ← texto plano, pronto para ingest
│   ├── ep002.txt
│   └── ...
└── audio/           ← só existe com --keep-audio
    ├── ep001.mp3
    └── ...
```

## Idempotência

- Se `data/transcripts/ep001.txt` já existe, o script **pula** esse episódio
- Se `data/audio/ep001.mp3` já existe, o download é **pulado** (só transcreve)
- Seguro para re-executar sem duplicar trabalho

## Troubleshooting

| Problema | Causa provável | Solução |
|----------|---------------|---------|
| `yt-dlp` não encontrado | Não instalado ou não no PATH | `pip install yt-dlp` |
| `ffmpeg` não encontrado | Não instalado | `sudo apt install ffmpeg` |
| "File too large" do Groq | Segmento > 25 MB | Reduza `MAX_SEGMENT_DURATION_SECONDS` |
| Rate limit Groq | Excedeu 14,400 seg/dia | Espere 24h ou use API paga |
| Áudio cortado/incompleto | yt-dlp falhou parcialmente | Delete o .mp3 e re-execute |
| Transcrição em idioma errado | Whisper confundiu o idioma | Já forçamos `language="en"` |

## Integração com ingest_episode.py

Após transcrever, rode o pipeline completo:

```bash
# Passo 1: Download + transcribe
python -m scripts.download_and_transcribe \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --episode-id ep001

# Passo 2: Ingest (clean → chunk → extract → resolve → upsert → embed)
python -m scripts.ingest_episode \
    --transcript data/transcripts/ep001.txt \
    --episode-id ep001 \
    --title "The Fall of the Roman Republic" \
    --number 1
```
