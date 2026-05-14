"""
Text Cleaner for Transcribed Audio.
====================================

PURPOSE:
    This is the FIRST step in the pipeline. It takes a raw transcription
    (from Groq/Whisper) and produces clean, consistent text ready for
    the chunker and extraction pipeline.

WHY IT EXISTS:
    Raw transcriptions are noisy. They contain:
    - Filler words ("uh", "uhm", "you know", "like")
    - Repetitions ("the the Roman Empire")
    - Inaudibility markers ("[inaudible]", "[unclear]")
    - Inconsistent whitespace (double spaces, random newlines)
    - Inline timestamps ("[00:15:32]")
    - Encoding artifacts (weird unicode characters)

    This noise:
    - Wastes tokens (we pay OpenAI per token — fillers are wasted money)
    - Confuses extraction (LLM sees "[inaudible] conquered [unclear]")
    - Breaks sentence splitting (false sentence boundaries)
    - Makes entity resolution harder (inconsistent name forms)

WHAT IT DOES:
    1. Normalizes whitespace and encoding
    2. Removes filler words and speech artifacts
    3. Removes transcription markers ([inaudible], timestamps, etc.)
    4. Removes stutters and immediate word repetitions
    5. Cleans up punctuation artifacts

WHAT IT DOES NOT DO:
    - Does NOT change semantic content (never alters what was said)
    - Does NOT correct factual errors from the transcriber
    - Does NOT translate anything
    - Does NOT remove "irrelevant" content (that's the extractor's job)
    - Does NOT do entity normalization (that's the resolver's job)

PIPELINE POSITION:
    Raw transcript → [text_cleaner.py] → Clean text → [chunker.py] → Chunks
"""

import re
import unicodedata


# =============================================================================
# FILLER WORDS
# =============================================================================
# These are common speech fillers in English that carry no semantic content.
# They appear constantly in transcribed lectures and waste tokens.
#
# We match them as WHOLE WORDS (using word boundaries \b) to avoid
# accidentally removing parts of real words.
# Example: "like" as filler vs. "like" in "something like a republic"
#
# Strategy: We only remove fillers that appear in specific patterns
# that are clearly non-semantic (e.g., at sentence starts, repeated,
# or surrounded by commas).
# =============================================================================

# Fillers that are ALWAYS safe to remove (never meaningful in lectures)
ALWAYS_REMOVE_FILLERS = [
    r'\buh\b',
    r'\buhm\b',
    r'\buhh\b',
    r'\bumm?\b',
    r'\bhmm?\b',
    r'\bahh?\b',
    r'\berr?\b',
]

# Fillers that are only removed when they appear in filler-like patterns
# (surrounded by commas, at sentence start, or repeated)
CONTEXTUAL_FILLERS = [
    r',\s*you know\s*,',       # "..., you know, ..." → "..., ..."
    r',\s*I mean\s*,',         # "..., I mean, ..." → "..., ..."
    r',\s*right\s*,',          # "..., right, ..." → "..., ..."
    r',\s*like\s*,',           # "..., like, ..." → "..., ..."
    r',\s*basically\s*,',      # "..., basically, ..." → "..., ..."
    r',\s*sort of\s*,',        # "..., sort of, ..." → "..., ..."
    r',\s*kind of\s*,',        # "..., kind of, ..." → "..., ..."
    r'^(So|And|But),?\s+',     # Sentence-starting "So, " / "And " (often filler in speech)
]

# Compile patterns for efficiency (these get used on every transcript)
_ALWAYS_REMOVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ALWAYS_REMOVE_FILLERS]
_CONTEXTUAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CONTEXTUAL_FILLERS]


# =============================================================================
# TRANSCRIPTION ARTIFACTS
# =============================================================================
# These are markers that transcription tools insert when they can't
# understand the audio, or metadata they embed in the text.
# =============================================================================

TRANSCRIPTION_ARTIFACTS = [
    r'\[inaudible\]',
    r'\[unclear\]',
    r'\[unintelligible\]',
    r'\[crosstalk\]',
    r'\[silence\]',
    r'\[music\]',
    r'\[applause\]',
    r'\[laughter\]',
    r'\(inaudible\)',
    r'\(unclear\)',
    r'\(unintelligible\)',
    # Inline timestamps: [00:15:32] or (00:15:32) or 00:15:32
    r'\[?\d{1,2}:\d{2}(:\d{2})?\]?',
    # Speaker labels that some transcribers add: "Speaker 1:" or "SPEAKER:"
    r'^(Speaker\s*\d*|SPEAKER\s*\d*)\s*:\s*',
]

_ARTIFACT_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in TRANSCRIPTION_ARTIFACTS]


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters to their standard forms.
    
    WHY: Transcription tools sometimes produce non-standard unicode:
    - "smart quotes" ('' "") → standard quotes (' ")
    - Em-dashes (—) → regular dashes (-)
    - Non-breaking spaces → regular spaces
    - Accented characters in various unicode forms → consistent NFC form
    
    We use NFC (Canonical Decomposition + Canonical Composition) which is
    the standard form for text processing.
    """
    # Normalize to NFC (standard composed form)
    text = unicodedata.normalize("NFC", text)

    # Replace smart quotes with standard quotes
    text = text.replace("\u2018", "'")   # left single quote
    text = text.replace("\u2019", "'")   # right single quote
    text = text.replace("\u201c", '"')   # left double quote
    text = text.replace("\u201d", '"')   # right double quote

    # Replace em-dash and en-dash with regular dash
    text = text.replace("\u2014", " - ")  # em-dash
    text = text.replace("\u2013", " - ")  # en-dash

    # Replace non-breaking space with regular space
    text = text.replace("\u00a0", " ")

    # Replace ellipsis character with three dots
    text = text.replace("\u2026", "...")

    return text


def remove_filler_words(text: str) -> str:
    """
    Remove speech filler words that carry no semantic content.
    
    TWO CATEGORIES:
    
    1. Always-remove fillers: "uh", "uhm", "hmm", etc.
       These are NEVER meaningful in a history lecture.
       We match them as whole words to avoid breaking real words.
    
    2. Contextual fillers: "you know", "I mean", "like", "basically"
       These CAN be meaningful ("I mean the Roman Empire" is valid).
       We only remove them when they appear in filler patterns:
       surrounded by commas, at sentence starts, etc.
    """
    # Remove always-safe fillers
    for pattern in _ALWAYS_REMOVE_PATTERNS:
        text = pattern.sub("", text)

    # Remove contextual fillers (only in filler-like positions)
    for pattern in _CONTEXTUAL_PATTERNS:
        # Replace with comma or space depending on context
        text = pattern.sub(", ", text)

    return text


def remove_transcription_artifacts(text: str) -> str:
    """
    Remove markers inserted by the transcription tool.
    
    These include:
    - [inaudible], [unclear], etc. — the transcriber couldn't understand
    - [music], [applause] — non-speech audio events
    - Inline timestamps — [00:15:32]
    - Speaker labels — "Speaker 1:" (we know it's Jiang)
    
    We remove these entirely because:
    - They carry no extractable knowledge
    - They waste tokens
    - They confuse the extraction LLM
    """
    for pattern in _ARTIFACT_PATTERNS:
        text = pattern.sub("", text)

    return text


def remove_stutters(text: str) -> str:
    """
    Remove immediate word repetitions (stutters).
    
    In speech, people often repeat words:
    - "the the Roman Empire" → "the Roman Empire"
    - "and and then" → "and then"
    - "he he conquered" → "he conquered"
    
    We detect consecutive identical words (case-insensitive) and keep only one.
    We only remove IMMEDIATE repetitions (adjacent words), not intentional
    repetition for emphasis across sentences.
    """
    # Match a word followed by the same word (case-insensitive)
    # \1 is a backreference to the first captured group
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    return text


def normalize_whitespace(text: str) -> str:
    """
    Clean up whitespace issues.
    
    After removing fillers and artifacts, we often end up with:
    - Double spaces: "the  Roman Empire"
    - Spaces before punctuation: "Rome , the capital"
    - Multiple commas: "Caesar,, the dictator"
    - Leading/trailing whitespace on lines
    - Multiple blank lines
    
    This function normalizes all of that to clean, consistent text.
    """
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)

    # Remove space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Remove multiple consecutive commas (artifact of filler removal)
    text = re.sub(r',(\s*,)+', ',', text)

    # Remove comma at start of sentence
    text = re.sub(r'^\s*,\s*', '', text, flags=re.MULTILINE)

    # Normalize line breaks: collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Strip the entire text
    text = text.strip()

    return text


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def clean_transcript(text: str) -> str:
    """
    Full cleaning pipeline. This is what you call from outside.
    
    Applies all cleaning steps in order:
    1. Unicode normalization (consistent character encoding)
    2. Remove transcription artifacts ([inaudible], timestamps, etc.)
    3. Remove filler words (uh, uhm, you know, etc.)
    4. Remove stutters (repeated words)
    5. Normalize whitespace (clean up the mess left by removals)
    
    The ORDER matters:
    - We normalize unicode FIRST so regex patterns match consistently
    - We remove artifacts BEFORE fillers (artifacts might contain filler-like text)
    - We normalize whitespace LAST to clean up gaps left by all removals
    
    Parameters:
        text: Raw transcript text from Groq/Whisper
    
    Returns:
        Cleaned text ready for the chunker
    """
    # Step 1: Make all characters consistent
    text = normalize_unicode(text)

    # Step 2: Remove transcription tool artifacts
    text = remove_transcription_artifacts(text)

    # Step 3: Remove speech fillers
    text = remove_filler_words(text)

    # Step 4: Remove stutters/repetitions
    text = remove_stutters(text)

    # Step 5: Clean up whitespace
    text = normalize_whitespace(text)

    return text
