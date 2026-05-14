"""
Two-Pass Chunking Strategy for History Class Transcripts.

Level 1 (Segment Detection via LLM): Uses a language model to identify natural topic
    boundaries in the transcript. The LLM understands semantic context and can detect
    topic shifts that no regex or embedding method would catch.

Level 2 (Token-Bounded Chunking): Deterministic splitting of segments into ~1500 token
    chunks with sentence-level overlap. No LLM needed here.

Architecture note:
    The LLM call for Level 1 is isolated in `segment_with_llm()`. This function
    requires an OpenAI API key to run. When no API is available (e.g., during
    development/testing), you can:
    - Pass pre-built segments directly to `chunk_segment()` for Level 2 testing
    - Use `segment_with_llm_mock()` which returns uniform segments for testing
    - Provide your own segmentation and skip Level 1 entirely

    The rest of the code (sentence splitting, segment construction, Level 2 chunking)
    is fully deterministic and has zero external dependencies.
"""

from dataclasses import dataclass
from pydantic import BaseModel, Field
import re


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Chunk:
    """
    A single chunk of text ready for extraction.
    Maps to an Utterance node in the Neo4j graph.
    """
    chunk_id: str               # Unique identifier: "ep{N}-seg{M}-chunk{K}"
    text: str                   # The chunk text
    episode_id: str             # Which episode this belongs to
    segment_index: int          # Which Level 1 segment this belongs to
    chunk_index: int            # Position within the segment
    global_index: int           # Position within the entire episode
    token_count: int            # Number of tokens (estimated)
    start_char: int             # Character offset in original text (start)
    end_char: int               # Character offset in original text (end)
    overlap_with_previous: bool # Whether this chunk overlaps with the previous one


@dataclass
class Segment:
    """
    A Level 1 segment — a coherent block of discussion on a single topic.
    Maps to a Segment node in the Neo4j graph.

    Created by the LLM segmentation pass, which identifies where the speaker
    transitions between topics.
    """
    segment_index: int          # Position within the episode
    sentences: list[str]        # The sentences in this segment
    topic: str                  # Brief topic description (from LLM)
    start_char: int             # Character offset in original text (start)
    end_char: int               # Character offset in original text (end)


# =============================================================================
# Pydantic schema for LLM segmentation output (used with Structured Outputs)
# =============================================================================


class TopicBoundary(BaseModel):
    """A single topic boundary identified by the LLM."""
    sentence_index: int = Field(
        description="The index of the sentence where a NEW topic begins (0-indexed)."
    )
    topic_description: str = Field(
        description="Brief description of the topic that starts at this sentence. "
        "E.g., 'The military campaigns of Alexander in Persia'."
    )


class SegmentationResult(BaseModel):
    """Output of the LLM segmentation call for a window of sentences."""
    boundaries: list[TopicBoundary] = Field(
        description="List of topic boundaries found in this window. "
        "Each boundary marks where a new topic begins. "
        "If no topic change is found, return an empty list."
    )


# =============================================================================
# LLM Segmentation Prompt
# =============================================================================

SEGMENTATION_SYSTEM_PROMPT = """You are analyzing a history lecture transcript to identify topic boundaries.

You will receive a numbered list of sentences from a lecture by Prof. Jiang (Predictive History YouTube channel). Your job is to identify where the speaker CHANGES TOPIC — i.e., transitions from discussing one subject to discussing a different subject.

WHAT COUNTS AS A TOPIC CHANGE:
- Moving from one historical period/event to another (e.g., from Roman Republic to Greek city-states)
- Shifting from historical narrative to modern geopolitical analysis
- Transitioning from one person/empire to another
- Moving from factual exposition to opinion/prediction (or vice versa)
- Starting a new causal argument or analytical thread

WHAT DOES NOT COUNT:
- Continuing to elaborate on the same topic with more detail
- Giving an example that illustrates the current topic
- A brief aside that returns to the main topic within 1-2 sentences
- Rhetorical questions that serve the current argument

RULES:
1. Be conservative — only mark boundaries where there is a CLEAR topic shift.
2. A segment should typically be 5-30 sentences. Avoid creating very short segments (< 5 sentences) unless the topic truly changes that quickly.
3. The first sentence (index 0) is always the start of a topic — do NOT include index 0 as a boundary (it's implicit).
4. Provide a brief topic_description for each new topic that begins.
5. If the entire window discusses a single topic, return an empty boundaries list."""

SEGMENTATION_USER_TEMPLATE = """SENTENCES:
{numbered_sentences}

Identify where topic changes occur in these sentences."""


# =============================================================================
# Token Counting
# =============================================================================


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a piece of text.

    Uses a heuristic of ~1.3 tokens per word for English text.
    This is a well-known approximation for GPT tokenizers.

    In production, for exact counts, replace with:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(enc.encode(text))

    The heuristic is conservative enough for chunking decisions.
    """
    word_count = len(text.split())
    return int(word_count * 1.3)


# =============================================================================
# Sentence Splitting (deterministic, no external dependencies)
# =============================================================================


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation boundaries.

    Handles common edge cases in transcribed speech:
    - Abbreviations (Dr., Mr., St., B.C., A.D., U.S., etc.)
    - Ellipsis (...)
    - Multiple punctuation (!!, ?!)

    Returns a list of non-empty sentences with whitespace stripped.
    """
    # Protect common abbreviations from being treated as sentence endings
    protected = text
    abbreviations = [
        "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St.", "Jr.", "Sr.",
        "vs.", "etc.", "i.e.", "e.g.", "B.C.", "A.D.", "U.S.", "U.K.",
        "Mt.", "Gen.", "Col.", "Sgt.", "Lt.", "Capt.",
    ]
    placeholders: dict[str, str] = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        placeholders[placeholder] = abbr
        protected = protected.replace(abbr, placeholder)

    # Split on sentence-ending punctuation followed by whitespace + uppercase letter
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)

    # Restore abbreviations and clean up
    sentences: list[str] = []
    for part in parts:
        for placeholder, abbr in placeholders.items():
            part = part.replace(placeholder, abbr)
        stripped = part.strip()
        if stripped:
            sentences.append(stripped)

    return sentences


# =============================================================================
# Level 1 — LLM-Based Segment Detection
# =============================================================================


def prepare_segmentation_windows(
    sentences: list[str],
    window_size: int = 20,
    window_overlap: int = 3,
) -> list[list[tuple[int, str]]]:
    """
    Prepare sliding windows of sentences for LLM segmentation.

    Each window is a list of (global_sentence_index, sentence_text) tuples.
    Windows overlap by `window_overlap` sentences to ensure boundaries
    at window edges are not missed.

    Parameters:
        sentences: All sentences from the transcript
        window_size: How many sentences per LLM call
        window_overlap: Overlap between consecutive windows

    Returns:
        List of windows, each containing (index, sentence) tuples
    """
    windows: list[list[tuple[int, str]]] = []
    i = 0

    while i < len(sentences):
        end = min(i + window_size, len(sentences))
        window = [(idx, sentences[idx]) for idx in range(i, end)]
        windows.append(window)

        # Advance by (window_size - overlap), ensuring progress
        i += window_size - window_overlap
        if i >= len(sentences):
            break

    return windows


def format_window_for_prompt(window: list[tuple[int, str]]) -> str:
    """
    Format a window of sentences as a numbered list for the LLM prompt.

    Example output:
        [0] Alexander the Great was born in 356 BC in Pella, Macedonia.
        [1] His father, Philip II, had already unified most of Greece.
        [2] Now, the young prince would inherit an empire ready for expansion.
    """
    lines = [f"[{idx}] {sentence}" for idx, sentence in window]
    return "\n".join(lines)


def build_segments_from_boundaries(
    sentences: list[str],
    boundaries: list[TopicBoundary],
    full_text: str,
) -> list[Segment]:
    """
    Construct Segment objects from the LLM-identified boundaries.

    This is the deterministic step that takes the LLM output (boundary indices)
    and builds the actual Segment data structures with text and metadata.

    Parameters:
        sentences: All sentences from the transcript
        boundaries: Topic boundaries identified by the LLM (sorted by sentence_index)
        full_text: The original full text (for character offset calculation)

    Returns:
        List of Segment objects ordered by segment_index
    """
    if not sentences:
        return []

    # Sort boundaries by sentence index and deduplicate
    sorted_boundaries = sorted(set(
        (b.sentence_index, b.topic_description) for b in boundaries
    ), key=lambda x: x[0])

    # Build boundary indices list (always starts at 0)
    boundary_indices: list[tuple[int, str]] = [(0, "Opening topic")]
    for idx, topic in sorted_boundaries:
        if idx > 0 and idx < len(sentences):  # Validate index
            boundary_indices.append((idx, topic))

    # Construct segments
    segments: list[Segment] = []

    for seg_i, (start_idx, topic) in enumerate(boundary_indices):
        # Determine end index (start of next segment, or end of sentences)
        if seg_i + 1 < len(boundary_indices):
            end_idx = boundary_indices[seg_i + 1][0]
        else:
            end_idx = len(sentences)

        segment_sentences = sentences[start_idx:end_idx]
        segment_text = " ".join(segment_sentences)

        # Calculate character offsets in the original text
        preceding_text = " ".join(sentences[:start_idx])
        start_char = len(preceding_text) + (1 if start_idx > 0 else 0)
        end_char = start_char + len(segment_text)

        segments.append(Segment(
            segment_index=seg_i,
            sentences=segment_sentences,
            topic=topic,
            start_char=start_char,
            end_char=end_char,
        ))

    return segments


async def segment_with_llm(
    sentences: list[str],
    full_text: str,
    llm_client,
    model: str = "gpt-4o-mini",
    window_size: int = 20,
    window_overlap: int = 3,
) -> list[Segment]:
    """
    Level 1: Use an LLM to detect topic boundaries in the transcript.

    THIS FUNCTION REQUIRES AN OPENAI API KEY.
    It calls the OpenAI API with structured outputs to get topic boundaries.

    Parameters:
        sentences: All sentences from the transcript (from split_into_sentences)
        full_text: The original full text
        llm_client: An initialized OpenAI client (openai.AsyncOpenAI)
        model: Which model to use (default: gpt-4o-mini for cost efficiency)
        window_size: Sentences per LLM call
        window_overlap: Overlap between windows

    Returns:
        List of Segment objects
    """
    # Prepare sliding windows
    windows = prepare_segmentation_windows(sentences, window_size, window_overlap)

    # Collect all boundaries across windows
    all_boundaries: list[TopicBoundary] = []

    for window in windows:
        # Format the window for the prompt
        numbered_text = format_window_for_prompt(window)
        user_prompt = SEGMENTATION_USER_TEMPLATE.format(
            numbered_sentences=numbered_text
        )

        # Call the LLM with structured output
        response = await llm_client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SEGMENTATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "SegmentationResult",
                    "strict": True,
                    "schema": SegmentationResult.model_json_schema(),
                }
            },
        )

        # Parse the response
        import json
        result_json = json.loads(response.choices[0].message.content)
        result = SegmentationResult(**result_json)

        # Map window-local indices back to global sentence indices
        for boundary in result.boundaries:
            # The window contains (global_index, sentence) tuples
            # boundary.sentence_index is relative to the window
            if boundary.sentence_index < len(window):
                global_idx = window[boundary.sentence_index][0]
                all_boundaries.append(TopicBoundary(
                    sentence_index=global_idx,
                    topic_description=boundary.topic_description,
                ))

    # Deduplicate boundaries (windows overlap, so same boundary may appear twice)
    seen_indices: set[int] = set()
    unique_boundaries: list[TopicBoundary] = []
    for b in sorted(all_boundaries, key=lambda x: x.sentence_index):
        if b.sentence_index not in seen_indices:
            seen_indices.add(b.sentence_index)
            unique_boundaries.append(b)

    # Build segments from boundaries
    return build_segments_from_boundaries(sentences, unique_boundaries, full_text)


def segment_with_llm_mock(
    sentences: list[str],
    full_text: str,
    sentences_per_segment: int = 15,
) -> list[Segment]:
    """
    Mock segmentation for testing without API access.

    Creates uniform segments of `sentences_per_segment` sentences each.
    Useful for testing the Level 2 chunking logic independently.

    Parameters:
        sentences: All sentences from the transcript
        full_text: The original full text
        sentences_per_segment: How many sentences per mock segment

    Returns:
        List of Segment objects (uniformly sized)
    """
    segments: list[Segment] = []
    seg_index = 0

    for i in range(0, len(sentences), sentences_per_segment):
        segment_sentences = sentences[i:i + sentences_per_segment]
        segment_text = " ".join(segment_sentences)

        preceding_text = " ".join(sentences[:i])
        start_char = len(preceding_text) + (1 if i > 0 else 0)
        end_char = start_char + len(segment_text)

        segments.append(Segment(
            segment_index=seg_index,
            sentences=segment_sentences,
            topic=f"Mock segment {seg_index}",
            start_char=start_char,
            end_char=end_char,
        ))
        seg_index += 1

    return segments


# =============================================================================
# Level 2 — Token-Bounded Chunking with Overlap (deterministic, no LLM)
# =============================================================================


def chunk_segment(
    segment: Segment,
    episode_id: str,
    global_index_start: int,
    max_chunk_tokens: int = 1500,
    min_chunk_tokens: int = 200,
    overlap_sentences: int = 3,
) -> list[Chunk]:
    """
    Level 2: Split a segment into token-bounded chunks with sentence-level overlap.

    This is fully deterministic — no LLM calls. It operates on the segments
    produced by Level 1 and splits them into extraction-ready chunks.

    Strategy:
    1. If the segment fits in one chunk (≤ max_chunk_tokens), return it as-is
    2. Otherwise, accumulate sentences until approaching max_chunk_tokens
    3. Close the chunk at the last complete sentence that fits
    4. Start the next chunk with `overlap_sentences` from the end of the previous
    5. Tiny trailing chunks (< min_chunk_tokens) are merged into the previous chunk

    Parameters:
        segment: A Level 1 segment to chunk
        episode_id: Episode identifier for the chunk_id
        global_index_start: Starting global index for chunks from this segment
        max_chunk_tokens: Maximum tokens per chunk
        min_chunk_tokens: Minimum tokens for a chunk to be independent
        overlap_sentences: Number of sentences to overlap between consecutive chunks

    Returns:
        List of Chunk objects
    """
    sentences = segment.sentences

    if not sentences:
        return []

    # If the entire segment fits in one chunk, return it as-is
    full_text = " ".join(sentences)
    full_tokens = estimate_tokens(full_text)

    if full_tokens <= max_chunk_tokens:
        return [Chunk(
            chunk_id=f"{episode_id}-seg{segment.segment_index:02d}-chunk00",
            text=full_text,
            episode_id=episode_id,
            segment_index=segment.segment_index,
            chunk_index=0,
            global_index=global_index_start,
            token_count=full_tokens,
            start_char=segment.start_char,
            end_char=segment.end_char,
            overlap_with_previous=False,
        )]

    # Split into multiple chunks with overlap
    chunks: list[Chunk] = []
    chunk_index = 0
    i = 0  # Current sentence index

    while i < len(sentences):
        # Accumulate sentences for this chunk
        chunk_sentences: list[str] = []
        token_count = 0
        j = i

        while j < len(sentences):
            sentence_tokens = estimate_tokens(sentences[j])

            # If adding this sentence exceeds the limit and we already have content, stop
            if token_count + sentence_tokens > max_chunk_tokens and chunk_sentences:
                break

            chunk_sentences.append(sentences[j])
            token_count += sentence_tokens
            j += 1

        # Build the chunk text
        chunk_text = " ".join(chunk_sentences)
        actual_tokens = estimate_tokens(chunk_text)

        # Calculate character offsets
        preceding_text = " ".join(sentences[:i])
        start_char = segment.start_char + len(preceding_text) + (1 if i > 0 else 0)
        end_char = start_char + len(chunk_text)

        chunks.append(Chunk(
            chunk_id=f"{episode_id}-seg{segment.segment_index:02d}-chunk{chunk_index:02d}",
            text=chunk_text,
            episode_id=episode_id,
            segment_index=segment.segment_index,
            chunk_index=chunk_index,
            global_index=global_index_start + chunk_index,
            token_count=actual_tokens,
            start_char=start_char,
            end_char=end_char,
            overlap_with_previous=(chunk_index > 0),
        ))

        chunk_index += 1

        # Advance: go back overlap_sentences from where we stopped
        next_start = j - overlap_sentences
        # Ensure we always make forward progress
        i = max(next_start, i + 1)

        # If we've consumed all sentences, stop
        if j >= len(sentences):
            break

    # Merge tiny trailing chunk into previous if too small
    if len(chunks) > 1 and chunks[-1].token_count < min_chunk_tokens:
        last = chunks.pop()
        prev = chunks[-1]
        # Append the non-overlapping portion of the last chunk
        prev_sentences = split_into_sentences(prev.text)
        last_sentences = split_into_sentences(last.text)

        # Find sentences in last that aren't already in prev
        new_sentences = [s for s in last_sentences if s not in prev_sentences[-overlap_sentences:]]
        if new_sentences:
            prev.text = prev.text + " " + " ".join(new_sentences)
        prev.token_count = estimate_tokens(prev.text)
        prev.end_char = last.end_char

    return chunks


# =============================================================================
# Main Entry Point — Full Pipeline
# =============================================================================


async def chunk_transcript(
    text: str,
    episode_id: str,
    llm_client=None,
    model: str = "gpt-4o-mini",
    max_chunk_tokens: int = 1500,
    min_chunk_tokens: int = 200,
    overlap_sentences: int = 3,
    window_size: int = 20,
    window_overlap: int = 3,
) -> list[Chunk]:
    """
    Full two-pass chunking pipeline.

    Level 1: LLM-based segmentation into topically coherent blocks.
    Level 2: Deterministic token-bounded chunking with overlap.

    If no llm_client is provided, falls back to mock segmentation
    (uniform segments for testing purposes).

    Parameters:
        text: The full cleaned transcript text
        episode_id: Identifier for the episode (e.g., "ep001")
        llm_client: An initialized OpenAI AsyncClient (or None for mock)
        model: Model for Level 1 segmentation
        max_chunk_tokens: Maximum tokens per Level 2 chunk
        min_chunk_tokens: Minimum tokens for a chunk to be independent
        overlap_sentences: Sentence overlap between consecutive chunks
        window_size: Sentences per LLM window in Level 1
        window_overlap: Overlap between LLM windows in Level 1

    Returns:
        List of Chunk objects, ordered by global_index
    """
    # Step 0: Split into sentences (deterministic)
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    # Step 1: Level 1 — Segment detection
    if llm_client is not None:
        segments = await segment_with_llm(
            sentences=sentences,
            full_text=text,
            llm_client=llm_client,
            model=model,
            window_size=window_size,
            window_overlap=window_overlap,
        )
    else:
        # No API available — use mock segmentation for testing
        segments = segment_with_llm_mock(sentences, text)

    # Step 2: Level 2 — Token-bounded chunking within each segment
    all_chunks: list[Chunk] = []
    global_index = 0

    for segment in segments:
        segment_chunks = chunk_segment(
            segment=segment,
            episode_id=episode_id,
            global_index_start=global_index,
            max_chunk_tokens=max_chunk_tokens,
            min_chunk_tokens=min_chunk_tokens,
            overlap_sentences=overlap_sentences,
        )
        all_chunks.extend(segment_chunks)
        global_index += len(segment_chunks)

    return all_chunks
