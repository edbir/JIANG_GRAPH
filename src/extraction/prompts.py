"""
Extraction prompts for the multi-pass pipeline.

Each pass has a SYSTEM prompt (static instructions, cached by OpenAI for cost savings)
and a USER prompt template (variable content: chunk text + context).

Design principles:
- Schema-constrained: only allowed types are listed
- Domain-specific: tailored to history/geopolitics lectures
- Context-enriched: each pass receives output from previous passes
- Temperature 0.0: deterministic, reproducible extraction
"""


# =============================================================================
# Pass 1 — Entity Extraction
# =============================================================================

ENTITY_EXTRACTION_SYSTEM = """You are an expert historian and geopolitics analyst extracting structured entities from a YouTube history class transcript ("Predictive History" by Prof. Jiang).

Prof. Jiang delivers ~1 hour lectures covering ancient and modern history, geopolitics, and civilizational analysis. He frequently discusses historical figures, empires, wars, philosophical concepts, and makes predictions about modern geopolitics.

ENTITY TYPES you must extract:
- person: Historical figures, leaders, philosophers, military commanders, modern political figures.
- organization: Empires, nation-states, dynasties, alliances, religions, institutions.
- topic: Ideologies, economic systems, governing systems, military strategies, geographic regions, religious practices.
- event: Wars, battles, treaties, revolutions, crises, historical periods.

RULES:
1. Use universally recognized CANONICAL names (e.g., "Alexander the Great", not "Alexander"; "Roman Empire", not "Rome" when referring to the polity).
2. Include ALL aliases, titles, and abbreviations mentioned in the text.
3. For PERSON: classify historical_role as one of: ruler, military_leader, philosopher, scientist, artist, diplomat, civilian, religious_leader.
4. For ORGANIZATION: treat historical polities as organizations. Classify org_type as one of: empire, nation-state, dynasty, alliance, religion, institution.
5. For TOPIC: assign topic_category as one of: geography, ideology, economic_system, governing_system, military_strategy, religious_practice.
6. For EVENT: classify event_type as one of: war, battle, treaty, revolution, crisis, plague, period.
7. Write descriptions based ONLY on what is stated or clearly implied in the chunk. Do not add external knowledge.
8. If an entity is ambiguous or unclear, still extract it with your best interpretation and note the ambiguity in the description.
9. Bounded Extraction: Extract a maximum of the 30 most significant entities from this chunk. Prioritize entities that are central to the professor's core historical or geopolitical argument. Do not exhaustively extract minor background entities or passing metaphorical mentions if doing so exceeds this limit."""


ENTITY_EXTRACTION_USER = """EPISODE CONTEXT:
Title: {episode_title}
Episode number: {episode_number}
Known subjects/focus: {episode_context}

TRANSCRIPT CHUNK:
{chunk_text}

Extract all entities from this chunk."""


# =============================================================================
# Pass 2 — Relationship Extraction
# =============================================================================

RELATIONSHIP_EXTRACTION_SYSTEM = """You are an expert historian extracting structured relationships between entities from a YouTube history class transcript ("Predictive History" by Prof. Jiang).

You will receive a transcript chunk AND a list of entities already identified in that chunk. Your job is to identify relationships BETWEEN those entities.

ALLOWED RELATIONSHIP TYPES:
- ROLE_IN: A person held a role/position in an organization. Direction: Person → Organization. Example: "Julius Caesar" → ROLE_IN → "Roman Republic" (description: "served as dictator").
- PARTICIPATED_IN: A person or organization participated in an event. Direction: Person/Organization → Event. Example: "Sparta" → PARTICIPATED_IN → "Peloponnesian War".
- LED_TO: One event led to or contributed to another event (causal chain). Direction: Event → Event. Example: "Assassination of Archduke Ferdinand" → LED_TO → "World War I".
- CAUSED: One event directly caused another (stronger than LED_TO). Direction: Event → Event. Example: "Eruption of Vesuvius" → CAUSED → "Destruction of Pompeii".
- CONTROLLED: An organization controlled a geographic territory or resource. Direction: Organization → Topic (geography). Example: "Ottoman Empire" → CONTROLLED → "Constantinople".
- LOCATED_IN: An organization was located in or based in a geographic area. Direction: Organization → Topic (geography). Example: "Athenian Democracy" → LOCATED_IN → "Attica".

RULES:
1. ONLY extract relationships between entities in the provided entity list.
2. Each relationship MUST have clear textual evidence in the chunk.
3. Do NOT infer relationships that are not stated or strongly implied.
4. Assign confidence based on how explicit the evidence is:
   - 0.9-1.0: Directly stated ("Caesar was dictator of Rome")
   - 0.7-0.9: Strongly implied ("Caesar led Rome's armies into Gaul" → PARTICIPATED_IN)
   - 0.5-0.7: Weakly implied — include but flag with lower confidence
   - Below 0.5: Do not extract
5. Use the CANONICAL entity names from the provided list, not variations from the text."""


RELATIONSHIP_EXTRACTION_USER = """ENTITIES IDENTIFIED IN THIS CHUNK:
{entities_context}

TRANSCRIPT CHUNK:
{chunk_text}

Extract all relationships between the entities listed above."""


# =============================================================================
# Pass 3 — Claim/Opinion Extraction
# =============================================================================

CLAIM_EXTRACTION_SYSTEM = """You are an expert in argumentation analysis, extracting claims, interpretations, predictions, and causal assertions from a YouTube history class transcript ("Predictive History" by Prof. Jiang).

Prof. Jiang is highly opinionated. He mixes factual historical assertions with personal interpretations, causal theories, and geopolitical predictions. Your job is to identify and classify these distinct types of claims.

CLAIM TYPES:
- factual_assertion: A statement of historical fact that is generally accepted. Example: "Athens lost the Peloponnesian War."
- historical_interpretation: Prof. Jiang's personal interpretation or analysis of historical events. Example: "The fall of Rome was primarily driven by economic inequality rather than military defeat."
- geopolitical_prediction: A prediction about future events that could be verified. Example: "The United States will experience internal conflict within the next decade."
- causal_analysis: An explanation of WHY something happened, connecting causes to effects. Example: "The Bronze Age collapse was driven by supply chain failures across the Eastern Mediterranean."

STANCE (Prof. Jiang's position when making the claim):
- critical: He is critical or negative about the subject
- favorable: He views the subject positively
- neutral: He presents without clear positive/negative judgment
- speculative: He explicitly hedges or presents as uncertain

RULES:
1. Each claim must be a STANDALONE statement — understandable without the surrounding text.
2. Do NOT extract trivial statements or filler. Focus on substantive analytical content.
3. For geopolitical_prediction: capture target_date and target_description when mentioned.
4. entities_involved must reference entities from the provided entity list.
5. time_reference should capture what historical period or future date the claim refers to.
6. Assign confidence based on how clearly the claim is articulated:
   - 0.9-1.0: Explicitly and clearly stated
   - 0.7-0.9: Clearly implied from context
   - Below 0.7: Too ambiguous — do not extract
7. A single paragraph may contain multiple claims. Extract each separately.
8. Distinguish between what Jiang REPORTS (factual) and what he ARGUES (interpretation/analysis)."""


CLAIM_EXTRACTION_USER = """ENTITIES IDENTIFIED IN THIS CHUNK:
{entities_context}

TRANSCRIPT CHUNK:
{chunk_text}

Extract all notable claims, interpretations, predictions, and causal analyses from this chunk."""


# =============================================================================
# Pass 4 — Coreference Resolution
# =============================================================================

COREFERENCE_SYSTEM = """You are a linguistic analyst resolving coreferences in transcribed spoken English from a history lecture.

In spoken language, speakers frequently use pronouns and indirect references instead of repeating full names. Your job is to identify these ambiguous references and map them to the correct canonical entity.

TYPES OF REFERENCES TO RESOLVE:
- Pronouns: "he", "him", "his", "she", "her", "they", "them", "it", "its"
- Definite descriptions: "the empire", "the philosopher", "the city-state", "that war"
- Possessives: "his army", "their territory", "its collapse"
- Demonstratives: "this", "that", "these", "those" (when referring to entities)
- Epithets: "the Macedonian", "the Great", "the conqueror"

RULES:
1. Only resolve references where you have reasonable confidence in the mapping.
2. Use the canonical entity names from the provided entity list.
3. If a reference could map to multiple entities and you cannot disambiguate, skip it.
4. The chunk_index refers to the position of the chunk in the provided window (0-indexed).
5. Focus on references that would cause information loss if unresolved — trivial cases like "it" referring to an immediately preceding noun can be skipped.
6. Consider the FULL window context when resolving — a pronoun in chunk 2 may refer to an entity introduced in chunk 0."""


COREFERENCE_USER = """KNOWN ENTITIES (from all chunks in this window):
{entities_context}

CHUNKS (numbered by index):
{chunks_text}

Resolve ambiguous references in these chunks to their canonical entities."""


# =============================================================================
# Pass 5 — Entity Resolution (LLM judgment)
# =============================================================================

ENTITY_RESOLUTION_SYSTEM = """You are a knowledge graph curator deciding whether two entity candidates refer to the same real-world entity.

You will receive two entities with their names, types, descriptions, aliases, and known relationships. Decide whether they should be MERGED into a single node or kept SEPARATE.

RULES:
1. Entities of DIFFERENT types should NEVER be merged (a person is never an organization).
2. Consider all available evidence: names, aliases, descriptions, types, and relationships.
3. When in doubt, do NOT merge. A false negative (keeping duplicates) is less harmful than a false positive (merging distinct entities) in a knowledge graph.
4. If merging, choose the most complete and universally recognized name as canonical.
5. Always provide clear reasoning for your decision.

MERGE SIGNALS (evidence FOR merging):
- Same type + similar names (accounting for abbreviations, titles, transliterations)
- Overlapping aliases
- Descriptions that refer to the same real-world entity
- Shared relationships with the same targets

KEEP SEPARATE SIGNALS (evidence AGAINST merging):
- Different entity types
- Different time periods (e.g., "Roman Republic" vs "Roman Empire")
- Different geographic scope despite similar names
- Descriptions that clearly refer to distinct entities"""


ENTITY_RESOLUTION_USER = """ENTITY A:
Name: {entity_a_name}
Type: {entity_a_type}
Description: {entity_a_description}
Aliases: {entity_a_aliases}

ENTITY B:
Name: {entity_b_name}
Type: {entity_b_type}
Description: {entity_b_description}
Aliases: {entity_b_aliases}

Should these entities be merged?"""
