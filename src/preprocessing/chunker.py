"""
Two-Pass Chunking Strategy for History Class Transcripts.
==========================================================

WHY THIS EXISTS:
    We have ~1 hour lecture transcripts (~15,000-20,000 tokens each).
    We need to feed these to LLMs for entity/relationship/claim extraction.
    But LLMs work best with focused, coherent text chunks of ~1500 tokens.
    
    The naive approach (cut every 1500 tokens) destroys meaning:
    - Cuts ideas in half
    - Separates causes from effects  
    - Loses pronoun references ("he" gets separated from "Alexander")
    
    Our solution: TWO PASSES.

THE TWO PASSES:
    Level 1 — SEGMENT DETECTION (uses LLM):
        "Where does the professor change topic?"
        Input:  Full transcript as numbered sentences
        Output: Segments (variable-size blocks, each about ONE topic)
        Method: Ask an LLM to identify topic boundaries
        
    Level 2 — TOKEN-BOUNDED CHUNKING (deterministic, no LLM):
        "Split each segment into extraction-ready pieces"
        Input:  One segment (may be too long for extraction)
        Output: Chunks of ≤1500 tokens with 3-sentence overlap
        Method: Accumulate sentences until token limit, then start new chunk

ARCHITECTURE NOTE:
    The LLM call (Level 1) is isolated in `segment_with_llm()`.
    Everything else is deterministic and works without any API.
    
    For testing without API access:
    - Use `segment_with_llm_mock()` which creates uniform segments
    - Or call `chunk_segment()` directly with hand-built segments
"""

from dataclasses import dataclass
from pydantic import BaseModel, Field
import re


# =============================================================================
# DATA STRUCTURES
# =============================================================================
# These are the objects that flow through the pipeline.
# They carry both the text AND metadata needed for the graph later.
# =============================================================================


@dataclass
class Chunk:
    """
    A single chunk of text ready for extraction.
    
    This is the FINAL output of the chunker — what gets sent to the
    extraction pipeline (Pass 1-5 in schemas.py).
    
    Later, each Chunk becomes an Utterance node in Neo4j:
        chunk_id    → utteranceId
        text        → text property
        token_count → tokenCount property
        (embedding is generated later and stored on the node)
    """
    chunk_id: str               # Unique ID: "ep001-seg02-chunk03"
    text: str                   # The actual text content
    episode_id: str             # Which episode (e.g., "ep001")
    segment_index: int          # Which segment within the episode (0, 1, 2...)
    chunk_index: int            # Position within its segment (0, 1, 2...)
    global_index: int           # Position across the ENTIRE episode (0, 1, 2... N)
    token_count: int            # Approximate token count
    start_char: int             # Where this chunk starts in the original text
    end_char: int               # Where this chunk ends in the original text
    overlap_with_previous: bool # True if first few sentences repeat from previous chunk


@dataclass
class Segment:
    """
    A Level 1 segment — one coherent topic block.
    
    Created by the LLM in Level 1. Represents a stretch of the lecture
    where the professor is discussing ONE topic (e.g., "Alexander's 
    conquest of Persia" or "The fall of the Roman Republic").
    
    Later, each Segment becomes a Segment node in Neo4j:
        segment_index → segmentId
        topic         → topic property
    """
    segment_index: int          # Position within the episode (0, 1, 2...)
    sentences: list[str]        # All sentences in this segment
    topic: str                  # Brief topic description (from LLM)
    start_char: int             # Character offset start in original text
    end_char: int               # Character offset end in original text


# =============================================================================
# PYDANTIC SCHEMA FOR LLM SEGMENTATION
# =============================================================================
# This schema is passed to OpenAI's structured outputs so the LLM returns
# exactly the format we need — a list of boundary indices with topic labels.
# =============================================================================


class TopicBoundary(BaseModel):
    """
    One topic boundary identified by the LLM.
    
    Example: If the professor talks about Rome from sentences 0-12,
    then switches to Greece at sentence 13, the LLM returns:
        TopicBoundary(sentence_index=13, topic_description="Greek city-states")
    """
    sentence_index: int = Field(
        description="The index of the sentence where a NEW topic begins (0-indexed)."
    )
    topic_description: str = Field(
        description="Brief description of the topic that starts at this sentence. "
        "E.g., 'The military campaigns of Alexander in Persia'."
    )


class SegmentationResult(BaseModel):
    """
    What the LLM returns for each window of sentences.
    
    May contain 0 boundaries (entire window is one topic),
    or multiple boundaries (several topic changes in the window).
    """
    boundaries: list[TopicBoundary] = Field(
        description="List of topic boundaries found in this window. "
        "Each boundary marks where a new topic begins. "
        "If no topic change is found, return an empty list."
    )


# =============================================================================
# LLM SEGMENTATION PROMPT
# =============================================================================
# This prompt tells the LLM how to identify topic boundaries.
# It lives here (not in prompts.py) because it's specific to chunking,
# not to the extraction pipeline.
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
# TOKEN COUNTING
# =============================================================================
# We need to know how many tokens a piece of text has so we don't exceed
# the ~1500 token limit per chunk. This is an approximation — good enough
# for chunking decisions. Exact counting would use tiktoken but adds a dependency.
# =============================================================================


def estimate_tokens(text: str) -> int:
    """
    Estimate how many tokens a text will use with GPT tokenizers.
    
    WHY ~1.3 tokens per word:
        English text averages about 1.3 tokens per word with GPT tokenizers.
        "Alexander" = 1 token, "the" = 1 token, "Constantinople" = 3 tokens.
        The 1.3 multiplier accounts for this variation.
    
    WHY an estimate is fine here:
        We use this to decide "does this chunk fit in 1500 tokens?"
        Being off by 5-10% doesn't matter — we're not hitting a hard API limit,
        just targeting a comfortable extraction window size.
    
    For exact counts (if needed later):
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(enc.encode(text))
    """
    word_count = len(text.split())
    return int(word_count * 1.3)


# =============================================================================
# SENTENCE SPLITTING
# =============================================================================
# The foundation of everything. We split the transcript into sentences first,
# then work with sentences as our atomic unit. We NEVER cut mid-sentence.
#
# Why sentences as the atomic unit:
# - They are natural meaning boundaries
# - The LLM can reason about "sentence 5 starts a new topic"
# - Overlap is clean: "repeat last 3 sentences" is well-defined
# =============================================================================


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation.
    
    CHALLENGE: Transcribed speech has edge cases:
    - Abbreviations with periods: "Dr. Jiang said..." (not a sentence end)
    - Historical dates: "In 44 B.C. Caesar was..." (not a sentence end)
    - Ellipsis: "And then..." (IS a sentence end in speech)
    
    APPROACH:
    1. Protect known abbreviations by replacing their periods with placeholders
    2. Split on: period/exclamation/question + space + uppercase letter
    3. Restore the abbreviations
    
    This is simple and works well for English lecture transcripts.
    It's not perfect (no NLP sentence splitter is), but errors are rare
    and don't significantly impact chunking quality.
    """
    # Step 1: Protect abbreviations from being treated as sentence endings
    # We temporarily replace "Dr." with "__ABBR0__" so the period isn't a split point
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

    # Step 2: Split on sentence-ending punctuation followed by space + uppercase
    # The regex means: "after a . or ! or ?, followed by whitespace, before a capital letter"
    # This catches: "...fell. The next..." but NOT "...Dr. Smith..."
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)

    # Step 3: Restore abbreviations and clean up empty strings
    sentences: list[str] = []
    for part in parts:
        for placeholder, abbr in placeholders.items():
            part = part.replace(placeholder, abbr)
        stripped = part.strip()
        if stripped:
            sentences.append(stripped)

    return sentences


# =============================================================================
# LEVEL 1 — LLM-BASED SEGMENT DETECTION
# =============================================================================
# This is where we ask the LLM: "where does the professor change topic?"
#
# FLOW:
# 1. Take all sentences from the transcript
# 2. Group them into overlapping "windows" of ~20 sentences
# 3. For each window, ask the LLM to identify topic boundaries
# 4. Collect all boundaries, deduplicate, and build Segment objects
#
# WHY WINDOWS (not the whole transcript at once):
# - A full transcript might be 500+ sentences — too much for one prompt
# - Smaller windows give the LLM focused context for better decisions
# - Windows overlap so boundaries at edges aren't missed
# =============================================================================


def prepare_segmentation_windows(
    sentences: list[str],
    window_size: int = 20,
    window_overlap: int = 3,
) -> list[list[tuple[int, str]]]:
    """
    Group sentences into overlapping windows for LLM processing.
    
    EXAMPLE with window_size=5, overlap=2, and 12 sentences:
        Window 0: sentences [0, 1, 2, 3, 4]
        Window 1: sentences [3, 4, 5, 6, 7]    ← overlaps with window 0
        Window 2: sentences [6, 7, 8, 9, 10]   ← overlaps with window 1
        Window 3: sentences [9, 10, 11]         ← last window (shorter)
    
    WHY OVERLAP:
        If a topic boundary happens at sentence 4, it might be hard to detect
        without seeing sentence 5 (the start of the new topic). The overlap
        ensures that boundary appears in BOTH window 0 and window 1, so the
        LLM has context on both sides.
    
    Each window entry is a tuple of (global_sentence_index, sentence_text).
    We need the global index to map the LLM's response back to the full transcript.
    """
    windows: list[list[tuple[int, str]]] = []
    i = 0

    while i < len(sentences):
        # Take `window_size` sentences starting at position i
        end = min(i + window_size, len(sentences))
        window = [(idx, sentences[idx]) for idx in range(i, end)]
        windows.append(window)

        # Move forward by (window_size - overlap)
        # This creates the overlap: the last `overlap` sentences of this window
        # will be the first `overlap` sentences of the next window
        i += window_size - window_overlap

        # Safety: if we've passed the end, stop
        if i >= len(sentences):
            break

    return windows


def format_window_for_prompt(window: list[tuple[int, str]]) -> str:
    """
    Format a window of sentences as a numbered list for the LLM.
    
    The LLM sees something like:
        [0] Alexander the Great was born in 356 BC in Pella, Macedonia.
        [1] His father, Philip II, had already unified most of Greece.
        [2] Now, the young prince would inherit an empire ready for expansion.
    
    The numbers let the LLM say "topic changes at index 2" unambiguously.
    
    NOTE: The indices shown to the LLM are LOCAL to the window (0, 1, 2...),
    not global. We map back to global indices after getting the response.
    """
    lines = [f"[{idx}] {sentence}" for idx, sentence in window]
    return "\n".join(lines)


def build_segments_from_boundaries(
    sentences: list[str],
    boundaries: list[TopicBoundary],
    full_text: str,
) -> list[Segment]:
    """
    Take the LLM's boundary decisions and construct actual Segment objects.
    
    This is the DETERMINISTIC step after the LLM call. It doesn't make any
    judgment calls — it just slices the sentence list at the indices the LLM chose.
    
    EXAMPLE:
        sentences = ["A", "B", "C", "D", "E", "F", "G", "H"]
        boundaries = [TopicBoundary(sentence_index=3, topic="Topic 2"),
                      TopicBoundary(sentence_index=6, topic="Topic 3")]
        
        Result:
            Segment 0: sentences ["A", "B", "C"],       topic="Opening topic"
            Segment 1: sentences ["D", "E", "F"],       topic="Topic 2"
            Segment 2: sentences ["G", "H"],            topic="Topic 3"
    
    The first segment always starts at index 0 with topic "Opening topic"
    (since the LLM is told NOT to include index 0 as a boundary — it's implicit).
    """
    if not sentences:
        return []

    # Sort boundaries by sentence index and remove duplicates
    sorted_boundaries = sorted(set(
        (b.sentence_index, b.topic_description) for b in boundaries
    ), key=lambda x: x[0])

    # The first segment always starts at 0
    # Then each boundary marks the start of a new segment
    boundary_indices: list[tuple[int, str]] = [(0, "Opening topic")]
    for idx, topic in sorted_boundaries:
        if idx > 0 and idx < len(sentences):  # Validate: must be in range
            boundary_indices.append((idx, topic))

    # Now slice the sentences at each boundary to create segments
    segments: list[Segment] = []

    for seg_i, (start_idx, topic) in enumerate(boundary_indices):
        # This segment goes from start_idx to the next boundary (or end)
        if seg_i + 1 < len(boundary_indices):
            end_idx = boundary_indices[seg_i + 1][0]
        else:
            end_idx = len(sentences)

        segment_sentences = sentences[start_idx:end_idx]
        segment_text = " ".join(segment_sentences)

        # Calculate where this segment sits in the original text (character offsets)
        # This is approximate but useful for provenance tracking
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
    
    ⚠️  THIS FUNCTION REQUIRES AN OPENAI API KEY.
    It makes real API calls. Without a key, use segment_with_llm_mock() instead.
    
    HOW IT WORKS (step by step):
    
    1. Split sentences into overlapping windows of ~20 sentences each
    2. For each window:
       a. Format sentences as a numbered list
       b. Send to LLM with the segmentation prompt
       c. LLM returns: "topic changes at indices [X, Y, Z]"
       d. Map those LOCAL indices back to GLOBAL sentence indices
    3. Collect all boundaries from all windows
    4. Deduplicate (because windows overlap, same boundary may appear twice)
    5. Build Segment objects from the final boundary list
    
    WHY ASYNC:
        OpenAI API calls are I/O-bound (waiting for network response).
        Using async lets us potentially parallelize window processing later.
        For now it's sequential, but the interface is ready for optimization.
    
    Parameters:
        sentences:      All sentences from the transcript
        full_text:      The original full text (for character offset calculation)
        llm_client:     An initialized openai.AsyncOpenAI client
        model:          Which model to use (gpt-4o-mini is cheap and sufficient)
        window_size:    How many sentences per LLM call
        window_overlap: Overlap between consecutive windows
    
    Returns:
        List of Segment objects (the Level 1 output)
    """
    # Step 1: Prepare windows
    windows = prepare_segmentation_windows(sentences, window_size, window_overlap)

    # Step 2: Process each window through the LLM
    all_boundaries: list[TopicBoundary] = []

    for window in windows:
        # 2a. Format for the prompt
        numbered_text = format_window_for_prompt(window)
        user_prompt = SEGMENTATION_USER_TEMPLATE.format(
            numbered_sentences=numbered_text
        )

        # 2b. Call the LLM with structured output
        # The response_format ensures the LLM returns valid SegmentationResult JSON
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

        # 2c. Parse the LLM's response into our Pydantic model
        import json
        result_json = json.loads(response.choices[0].message.content)
        result = SegmentationResult(**result_json)

        # 2d. Map LOCAL window indices → GLOBAL sentence indices
        # The LLM says "boundary at index 7" meaning the 7th sentence IN THE WINDOW.
        # But we need to know which sentence that is in the FULL transcript.
        # The window stores (global_index, text) tuples, so we look up the global index.
        for boundary in result.boundaries:
            if boundary.sentence_index < len(window):
                global_idx = window[boundary.sentence_index][0]
                all_boundaries.append(TopicBoundary(
                    sentence_index=global_idx,
                    topic_description=boundary.topic_description,
                ))

    # Step 3: Deduplicate boundaries
    # Because windows overlap, the same boundary might be detected in two windows.
    # We keep only unique sentence indices.
    seen_indices: set[int] = set()
    unique_boundaries: list[TopicBoundary] = []
    for b in sorted(all_boundaries, key=lambda x: x.sentence_index):
        if b.sentence_index not in seen_indices:
            seen_indices.add(b.sentence_index)
            unique_boundaries.append(b)

    # Step 4: Build Segment objects from the deduplicated boundaries
    return build_segments_from_boundaries(sentences, unique_boundaries, full_text)


def segment_with_llm_mock(
    sentences: list[str],
    full_text: str,
    sentences_per_segment: int = 15,
) -> list[Segment]:
    """
    Mock segmentation for testing WITHOUT API access.
    
    Creates uniform segments of N sentences each. Obviously this doesn't
    respect actual topic boundaries — it's purely for testing the Level 2
    chunking logic without needing an API key.
    
    Use this when:
    - Developing/testing the Level 2 chunking code
    - Running the pipeline locally without API access
    - Writing unit tests
    
    In production, this is replaced by segment_with_llm().
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
# LEVEL 2 — TOKEN-BOUNDED CHUNKING WITH OVERLAP
# =============================================================================
# This is the DETERMINISTIC part. No LLM needed.
#
# It takes a Segment (from Level 1) and splits it into chunks that are:
# - At most ~1500 tokens (so they fit comfortably in extraction prompts)
# - Never cut mid-sentence (sentences are atomic)
# - Overlapping by 3 sentences (so cross-boundary info isn't lost)
#
# VISUAL EXAMPLE (overlap_sentences=2):
#
#   Segment sentences: [A, B, C, D, E, F, G, H, I, J, K, L]
#   Suppose max_chunk_tokens fits ~5 sentences per chunk.
#
#   Chunk 0: [A, B, C, D, E]           ← first chunk, no overlap
#   Chunk 1: [D, E, F, G, H]           ← starts 2 sentences back (overlap)
#   Chunk 2: [G, H, I, J, K]           ← starts 2 sentences back (overlap)
#   Chunk 3: [J, K, L]                 ← last chunk (shorter, that's ok)
#
#   Notice: sentences D,E appear in both chunk 0 and 1.
#   If a relationship spans sentences D→F, it's captured in chunk 1.
#   Without overlap, it would be split across chunks and potentially lost.
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
    Level 2: Split a segment into token-bounded chunks with overlap.
    
    FULLY DETERMINISTIC — no LLM calls, no external dependencies.
    
    ALGORITHM:
    1. Check if the whole segment fits in one chunk. If yes, return it as-is.
    2. Otherwise, start accumulating sentences:
       a. Add sentences one by one, counting tokens
       b. When adding the next sentence would exceed max_chunk_tokens, STOP
       c. Save the accumulated sentences as a chunk
       d. Start the next chunk `overlap_sentences` back from where we stopped
       e. Repeat until all sentences are consumed
    3. If the last chunk is tiny (< min_chunk_tokens), merge it into the previous
    
    Parameters:
        segment:            The Level 1 segment to split
        episode_id:         Episode ID for naming (e.g., "ep001")
        global_index_start: What global_index to start numbering from
        max_chunk_tokens:   Maximum tokens per chunk (default 1500)
        min_chunk_tokens:   Minimum tokens for a chunk to stand alone (default 200)
        overlap_sentences:  How many sentences to repeat between chunks (default 3)
    
    Returns:
        List of Chunk objects (may be just 1 if segment is small enough)
    """
    sentences = segment.sentences

    if not sentences:
        return []

    # --- CASE 1: Segment fits in one chunk → return as-is, no splitting needed ---
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

    # --- CASE 2: Segment is too long → split with overlap ---
    chunks: list[Chunk] = []
    chunk_index = 0
    i = 0  # Index of the first sentence in the current chunk

    while i < len(sentences):
        # ACCUMULATE sentences until we hit the token limit
        chunk_sentences: list[str] = []
        token_count = 0
        j = i  # j will advance through sentences

        while j < len(sentences):
            sentence_tokens = estimate_tokens(sentences[j])

            # Would adding this sentence exceed the limit?
            if token_count + sentence_tokens > max_chunk_tokens and chunk_sentences:
                # Yes, and we already have content → stop here
                break

            # No (or this is the first sentence) → add it
            chunk_sentences.append(sentences[j])
            token_count += sentence_tokens
            j += 1

        # BUILD the chunk from accumulated sentences
        chunk_text = " ".join(chunk_sentences)
        actual_tokens = estimate_tokens(chunk_text)

        # Calculate character offsets (approximate, for provenance)
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
            overlap_with_previous=(chunk_index > 0),  # First chunk has no overlap
        ))

        chunk_index += 1

        # ADVANCE with overlap:
        # Instead of starting the next chunk at j (where we stopped),
        # we go BACK by overlap_sentences. This creates the overlap.
        # 
        # Example: if j=10 and overlap=3, next chunk starts at sentence 7.
        # Sentences 7, 8, 9 will appear in BOTH this chunk and the next.
        next_start = j - overlap_sentences

        # Safety: always move forward by at least 1 sentence to avoid infinite loop
        i = max(next_start, i + 1)

        # If we've consumed all sentences, we're done
        if j >= len(sentences):
            break

    # --- CLEANUP: Merge tiny trailing chunk into previous ---
    # If the last chunk is very small (< min_chunk_tokens), it probably doesn't
    # have enough context for meaningful extraction. Merge it into the previous chunk.
    if len(chunks) > 1 and chunks[-1].token_count < min_chunk_tokens:
        last = chunks.pop()
        prev = chunks[-1]

        # Find sentences in the last chunk that aren't already in the previous
        # (because of overlap, some sentences are shared)
        prev_sentences = split_into_sentences(prev.text)
        last_sentences = split_into_sentences(last.text)
        new_sentences = [s for s in last_sentences if s not in prev_sentences[-overlap_sentences:]]

        if new_sentences:
            prev.text = prev.text + " " + " ".join(new_sentences)
        prev.token_count = estimate_tokens(prev.text)
        prev.end_char = last.end_char

    return chunks


# =============================================================================
# MAIN ENTRY POINT — FULL PIPELINE
# =============================================================================
# This is what you call from outside. It orchestrates both levels.
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
    Full two-pass chunking pipeline. This is the main entry point.
    
    WHAT IT DOES:
    1. Splits the transcript into sentences (deterministic)
    2. Level 1: Asks LLM to identify topic boundaries → produces Segments
    3. Level 2: Splits each Segment into token-bounded Chunks with overlap
    
    FALLBACK:
    If llm_client is None (no API available), uses mock segmentation.
    This creates uniform segments — not semantically meaningful, but allows
    testing the Level 2 logic without an API key.
    
    Parameters:
        text:              The full cleaned transcript text
        episode_id:        Episode identifier (e.g., "ep001")
        llm_client:        An openai.AsyncOpenAI instance, or None for mock mode
        model:             Model for Level 1 segmentation (default: gpt-4o-mini)
        max_chunk_tokens:  Max tokens per Level 2 chunk (default: 1500)
        min_chunk_tokens:  Min tokens for a chunk to be independent (default: 200)
        overlap_sentences: Sentence overlap between chunks (default: 3)
        window_size:       Sentences per LLM window in Level 1 (default: 20)
        window_overlap:    Overlap between LLM windows in Level 1 (default: 3)
    
    Returns:
        List of Chunk objects, ordered by global_index (0, 1, 2, ... N)
        Ready to be fed into the extraction pipeline.
    """
    # --- Step 0: Split into sentences ---
    # This is the foundation. Everything else works with sentences as atoms.
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    # --- Step 1: Level 1 — Segment detection ---
    if llm_client is not None:
        # Real mode: use LLM to find topic boundaries
        segments = await segment_with_llm(
            sentences=sentences,
            full_text=text,
            llm_client=llm_client,
            model=model,
            window_size=window_size,
            window_overlap=window_overlap,
        )
    else:
        # Mock mode: uniform segments for testing without API
        segments = segment_with_llm_mock(sentences, text)

    # --- Step 2: Level 2 — Token-bounded chunking ---
    # Process each segment independently. Overlap happens WITHIN segments,
    # not across segment boundaries (because segments are different topics).
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
