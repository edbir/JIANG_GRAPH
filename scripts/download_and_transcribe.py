"""
Download and Transcribe — YouTube Audio → Transcript Text File.
================================================================

THIS IS THE FIRST STEP in the pipeline. It takes a YouTube video URL,
downloads the audio using yt-dlp, transcribes it using Groq's Whisper API,
and saves the result as a plain text file ready for ingest_episode.py.

FLOW:
    YouTube URL → yt-dlp (download audio as mp3) → Groq Whisper API → transcript.txt

WHY GROQ:
    - Free tier available (enough for MVP)
    - Whisper large-v3 model (best open-source transcription)
    - Fast inference (Groq's LPU hardware)
    - English transcription quality is excellent for lecture content

WHY YT-DLP:
    - Best maintained YouTube downloader (yt-dlp fork of youtube-dl)
    - Handles age-restricted videos, playlists, channels
    - Can extract audio-only (saves bandwidth and time)
    - Handles format selection automatically

USAGE:
    # Single video
    python -m scripts.download_and_transcribe \
        --url "https://www.youtube.com/watch?v=VIDEO_ID" \
        --episode-id ep001

    # With custom output directory
    python -m scripts.download_and_transcribe \
        --url "https://www.youtube.com/watch?v=VIDEO_ID" \
        --episode-id ep001 \
        --output-dir data/transcripts

    # Keep the audio file after transcription
    python -m scripts.download_and_transcribe \
        --url "https://www.youtube.com/watch?v=VIDEO_ID" \
        --episode-id ep001 \
        --keep-audio

    # Batch: process multiple URLs from a file
    python -m scripts.download_and_transcribe \
        --batch episodes.txt

BATCH FILE FORMAT (episodes.txt):
    ep001 https://www.youtube.com/watch?v=VIDEO_ID_1 The Fall of Rome
    ep002 https://www.youtube.com/watch?v=VIDEO_ID_2 Alexander's Conquests
    ep003 https://www.youtube.com/watch?v=VIDEO_ID_3 The Bronze Age Collapse

PREREQUISITES:
    - GROQ_API_KEY in .env
    - yt-dlp installed (pip install yt-dlp)
    - ffmpeg installed (for audio conversion — yt-dlp uses it)
      On Ubuntu/Debian: sudo apt install ffmpeg
      On macOS: brew install ffmpeg

OUTPUT:
    data/transcripts/ep001.txt   — Plain text transcript (UTF-8)
    data/audio/ep001.mp3         — Audio file (only if --keep-audio)

COST:
    - yt-dlp download: FREE
    - Groq Whisper transcription: FREE tier (14,400 audio-seconds/day)
      That's ~4 hours/day — enough for 4 episodes per day.
      For 134 episodes at ~1 hour each: ~34 days on free tier.

GROQ WHISPER LIMITS (free tier):
    - Max file size: 25 MB
    - Max audio length: ~25 minutes per request
    - Daily limit: 14,400 audio-seconds
    
    For ~1 hour lectures, we split the audio into ~20-minute segments,
    transcribe each, and concatenate the results.
"""

import asyncio
import argparse
import subprocess
import tempfile
import math
from pathlib import Path

from groq import Groq

from src.config import settings


# =============================================================================
# CONSTANTS
# =============================================================================

AUDIO_DIR = Path("data/audio")
TRANSCRIPT_DIR = Path("data/transcripts")

# Groq Whisper limits
MAX_SEGMENT_DURATION_SECONDS = 1200  # 20 minutes per segment (safe under 25MB)
SUPPORTED_AUDIO_FORMAT = "mp3"


# =============================================================================
# YOUTUBE DOWNLOAD
# =============================================================================


def download_audio(url: str, episode_id: str, output_dir: Path = None) -> Path:
    """
    Download audio from a YouTube video using yt-dlp.
    
    Downloads as mp3 at 64kbps (sufficient for speech, keeps file small).
    A 1-hour lecture at 64kbps ≈ 29 MB.
    
    Parameters:
        url:        YouTube video URL
        episode_id: Episode identifier (used for filename)
        output_dir: Where to save the audio (default: data/audio/)
    
    Returns:
        Path to the downloaded audio file
    """
    if output_dir is None:
        output_dir = AUDIO_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{episode_id}.{SUPPORTED_AUDIO_FORMAT}"

    # If already downloaded, skip
    if output_path.exists():
        print(f"[download] Audio already exists: {output_path}")
        return output_path

    print(f"[download] Downloading audio from: {url}")
    print(f"[download] Output: {output_path}")

    # yt-dlp command:
    # -x              : extract audio only
    # --audio-format  : convert to mp3
    # --audio-quality : 64kbps (speech doesn't need more)
    # -o              : output filename template
    # --no-playlist   : don't download entire playlist if URL is from one
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", SUPPORTED_AUDIO_FORMAT,
        "--audio-quality", "64K",
        "-o", str(output_path),
        "--no-playlist",
        "--no-overwrites",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit code {result.returncode}):\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    # yt-dlp might add an extension, find the actual file
    if not output_path.exists():
        # Sometimes yt-dlp creates file with slightly different name
        candidates = list(output_dir.glob(f"{episode_id}.*"))
        if candidates:
            output_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Download completed but file not found at {output_path}. "
                f"Check {output_dir} for the actual file."
            )

    print(f"[download] ✓ Downloaded: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


# =============================================================================
# AUDIO DURATION
# =============================================================================


def get_audio_duration(audio_path: Path) -> float:
    """
    Get the duration of an audio file in seconds using ffprobe.
    
    Returns:
        Duration in seconds (float)
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    return float(result.stdout.strip())


# =============================================================================
# AUDIO SPLITTING
# =============================================================================


def split_audio(audio_path: Path, segment_duration: int = None) -> list[Path]:
    """
    Split a long audio file into segments for Groq's size limit.
    
    Uses ffmpeg to split without re-encoding (fast, lossless split).
    
    Parameters:
        audio_path:       Path to the audio file
        segment_duration: Max seconds per segment (default: 1200 = 20 min)
    
    Returns:
        List of paths to segment files (or just [audio_path] if short enough)
    """
    if segment_duration is None:
        segment_duration = MAX_SEGMENT_DURATION_SECONDS

    duration = get_audio_duration(audio_path)

    # If short enough, no splitting needed
    if duration <= segment_duration:
        return [audio_path]

    num_segments = math.ceil(duration / segment_duration)
    print(f"[split] Audio is {duration:.0f}s ({duration/60:.1f} min), splitting into {num_segments} segments")

    segments: list[Path] = []
    temp_dir = audio_path.parent / "segments"
    temp_dir.mkdir(exist_ok=True)

    for i in range(num_segments):
        start_time = i * segment_duration
        segment_path = temp_dir / f"{audio_path.stem}_seg{i:02d}.{SUPPORTED_AUDIO_FORMAT}"

        if segment_path.exists():
            segments.append(segment_path)
            continue

        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-ss", str(start_time),
            "-t", str(segment_duration),
            "-acodec", "libmp3lame",
            "-ab", "64k",
            "-y",  # overwrite
            "-loglevel", "quiet",
            str(segment_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg split failed for segment {i}: {result.stderr}")

        segments.append(segment_path)

    print(f"[split] ✓ Created {len(segments)} segments")
    return segments


# =============================================================================
# TRANSCRIPTION (GROQ WHISPER)
# =============================================================================


def transcribe_audio(audio_path: Path) -> str:
    """
    Transcribe a single audio file using Groq's Whisper API.
    
    Parameters:
        audio_path: Path to the audio file (must be < 25 MB)
    
    Returns:
        Transcription text (string)
    """
    client = Groq(api_key=settings.groq_api_key)

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path.name, audio_file.read()),
            model="whisper-large-v3",
            language="en",
            response_format="text",
        )

    return transcription


def transcribe_full_audio(audio_path: Path) -> str:
    """
    Transcribe a full audio file, splitting if necessary.
    
    Handles the Groq 25MB/25min limit by splitting long files
    into segments, transcribing each, and concatenating.
    
    Parameters:
        audio_path: Path to the audio file
    
    Returns:
        Full transcription text
    """
    # Split if needed
    segments = split_audio(audio_path)

    # Transcribe each segment
    transcriptions: list[str] = []
    for i, segment_path in enumerate(segments):
        print(f"[transcribe] Segment {i+1}/{len(segments)}: {segment_path.name}")
        text = transcribe_audio(segment_path)
        transcriptions.append(text.strip())
        print(f"[transcribe]   → {len(text)} chars")

    # Concatenate with double newline between segments
    full_text = "\n\n".join(transcriptions)

    # Clean up segment files (keep only the original audio)
    if len(segments) > 1:
        for seg in segments:
            if seg != audio_path:
                seg.unlink(missing_ok=True)
        # Remove segments directory if empty
        segments_dir = audio_path.parent / "segments"
        if segments_dir.exists() and not list(segments_dir.iterdir()):
            segments_dir.rmdir()

    return full_text


# =============================================================================
# FULL PIPELINE: DOWNLOAD + TRANSCRIBE
# =============================================================================


def download_and_transcribe(
    url: str,
    episode_id: str,
    output_dir: Path = None,
    keep_audio: bool = False,
) -> Path:
    """
    Full pipeline: download audio from YouTube and transcribe it.
    
    Parameters:
        url:         YouTube video URL
        episode_id:  Episode identifier
        output_dir:  Where to save the transcript (default: data/transcripts/)
        keep_audio:  Whether to keep the audio file after transcription
    
    Returns:
        Path to the saved transcript file
    """
    if output_dir is None:
        output_dir = TRANSCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = output_dir / f"{episode_id}.txt"

    # If transcript already exists, skip everything
    if transcript_path.exists():
        print(f"[pipeline] Transcript already exists: {transcript_path}")
        return transcript_path

    # Step 1: Download audio
    audio_path = download_audio(url, episode_id)

    # Step 2: Transcribe
    print(f"[transcribe] Starting transcription of {audio_path.name}...")
    transcript_text = transcribe_full_audio(audio_path)
    print(f"[transcribe] ✓ Transcription complete: {len(transcript_text)} chars")

    # Step 3: Save transcript
    transcript_path.write_text(transcript_text, encoding="utf-8")
    print(f"[pipeline] ✓ Saved transcript: {transcript_path}")

    # Step 4: Clean up audio (unless --keep-audio)
    if not keep_audio:
        audio_path.unlink(missing_ok=True)
        print(f"[pipeline] Removed audio file (use --keep-audio to retain)")

    return transcript_path


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def process_batch(batch_file: str, output_dir: Path = None, keep_audio: bool = False) -> list[Path]:
    """
    Process multiple episodes from a batch file.
    
    BATCH FILE FORMAT (one episode per line, tab or space separated):
        episode_id  youtube_url  optional_title
    
    Lines starting with # are comments. Empty lines are skipped.
    
    Example:
        ep001 https://www.youtube.com/watch?v=ABC123 The Fall of Rome
        ep002 https://www.youtube.com/watch?v=DEF456 Alexander's Conquests
    """
    batch_path = Path(batch_file)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")

    results: list[Path] = []
    lines = batch_path.read_text().strip().split("\n")

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=2)
        if len(parts) < 2:
            print(f"[batch] Skipping line {line_num}: insufficient fields")
            continue

        episode_id = parts[0]
        url = parts[1]
        title = parts[2] if len(parts) > 2 else ""

        print(f"\n{'='*60}")
        print(f"[batch] Processing {episode_id}: {title or url}")
        print(f"{'='*60}")

        try:
            transcript_path = download_and_transcribe(
                url=url,
                episode_id=episode_id,
                output_dir=output_dir,
                keep_audio=keep_audio,
            )
            results.append(transcript_path)
        except Exception as e:
            print(f"[batch] ✗ Failed {episode_id}: {e}")
            continue

    print(f"\n[batch] Done! {len(results)}/{len([l for l in lines if l.strip() and not l.startswith('#')])} episodes processed")
    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """Command-line interface for download and transcription."""
    parser = argparse.ArgumentParser(
        description="Download YouTube audio and transcribe using Groq Whisper."
    )

    # Mode: single video or batch
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url",
        help="YouTube video URL (single video mode)"
    )
    group.add_argument(
        "--batch",
        help="Path to batch file with episode_id, URL, and title per line"
    )

    # Required for single mode
    parser.add_argument(
        "--episode-id",
        help="Episode identifier (required for --url mode, e.g., 'ep001')"
    )

    # Optional
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for transcripts (default: data/transcripts/)"
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the downloaded audio file after transcription"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.url:
        # Single video mode
        if not args.episode_id:
            parser.error("--episode-id is required when using --url")

        download_and_transcribe(
            url=args.url,
            episode_id=args.episode_id,
            output_dir=output_dir,
            keep_audio=args.keep_audio,
        )
    else:
        # Batch mode
        process_batch(
            batch_file=args.batch,
            output_dir=output_dir,
            keep_audio=args.keep_audio,
        )


if __name__ == "__main__":
    main()
