# Operational Coding Grid for CamiHawke Comment Personas

## Purpose

This grid is meant to classify comments from `ig_comments_clean.csv` with a stable set of personas that works for:

- manual coding
- prompt design
- LLM classification

Each comment receives:

- one `primary_persona`
- `other_unclear` only when evidence remains insufficient

This is not a demographic taxonomy. It classifies the **dominant communicative function** of the comment.

## Unit of analysis

- unit: single comment
- minimum input: comment text
- ideal input: comment text + caption + reply status

Use the caption to resolve short, referential, or caption-dependent comments. If the caption makes the meaning clear, classify the comment.

## Final label set

- `affective`
- `identification`
- `humorous`
- `aesthetic`
- `informational`
- `validating`
- `argumentative`
- `autobiographical`
- `advisory`
- `situated_experts`
- `defensive`
- `sharing`
- `social_circle`
- `other_unclear`

## General assignment rule

Assign the label that best captures the main communicative act of the comment:

- warmth or admiration toward Cami, including brief praise with heart-eyes emojis -> `affective`
- self-recognition -> `identification`
- joke or amused reaction -> `humorous`
- compliment on looks or visual scene -> `aesthetic`
- request for concrete information -> `informational`
- endorsement of a point or message -> `validating`
- explicit opinion or reasoning -> `argumentative`
- personal story -> `autobiographical`
- recommendation or advice -> `advisory`
- role-based expertise -> `situated_experts`
- defense of Cami or discussion boundaries -> `defensive`
- tag, summon, or share-to-friend move -> `sharing`
- insider lore or repeated familiarity with Cami -> `social_circle`

## Recommended precedence order

If multiple categories seem plausible, use this order:

1. `social_circle`
2. `sharing`
3. `situated_experts`
4. `defensive`
5. `advisory`
6. `informational`
7. `autobiographical`
8. `argumentative`
9. `validating`
10. `affective`
11. `identification`
12. `aesthetic`
13. `humorous`
14. `other_unclear`

## Operational definitions

### 1. `affective`

Definition:
affection, warmth, symbolic closeness, or admiration of Cami as a person or relational figure. This also includes brief compliments marked by heart-eyes or "in love" emojis.

Positive indicators:
- `ti voglio bene`
- `adoro`
- `meravigliosa`
- `fantastica`
- `bona` or `figa` when paired with heart-eyes emojis
- `va bhe ciaone` when paired with heart-eyes emojis
- heart-eyes-only comments
- hearts used as warmth or closeness
- short praise not clearly about looks

Do not use if:
- the praise is clearly about appearance or visual scene and does not rely on heart-eyes admiration -> `aesthetic`
- the main act is endorsing a point -> `validating`

### 2. `identification`

Definition:
mirroring, self-recognition, or perceived similarity.

Positive indicators:
- `same`
- `idem`
- `anche io`
- `sono io`
- `questa sono io`

Do not use if:
- the comment develops into a personal narrative -> `autobiographical`

### 3. `humorous`

Definition:
laughter, irony, playful amusement, or caption-dependent comic reaction.

Positive indicators:
- `ahah`
- `muoio`
- punchline-like response
- short ironic line made clear by the caption
- emoji or phrasing that clearly signal comic intent

Do not use if:
- the main act is simply praise -> `affective` or `aesthetic`

### 4. `aesthetic`

Definition:
compliment on appearance, attractiveness, styling, image, or the visual scene when the main focus is visual rather than affective admiration.

Positive indicators:
- `bellissima`
- `stupenda`
- `ma quanto sei bona`
- `come sei bella`
- `che spettacolo` when the visual target is clear
- comments focused on how Cami looks

Do not use if:
- the compliment is brief and paired with heart-eyes or "in love" emojis -> `affective`
- the comment is mainly warmth or symbolic closeness -> `affective`

### 5. `informational`

Definition:
request for practical information, names, links, titles, or concrete references.

Positive indicators:
- `titolo?`
- `che libro e'?`
- `dove l'hai preso?`
- `link?`
- `che canzone e'?`

### 6. `validating`

Definition:
approval or endorsement of a point, message, reflection, or meaningful statement.

Positive indicators:
- `condivido`
- `esatto`
- `grazie per averlo detto`
- `hai ragione`
- support for the message more than for Cami as person

Do not use if:
- the comment is mostly praise of Cami -> `affective`

### 7. `argumentative`

Definition:
explicit opinion, thesis, disagreement, or developed reasoning.

Positive indicators:
- `secondo me`
- `penso che`
- `non sono d'accordo`
- `credo che`

### 8. `autobiographical`

Definition:
personal story, memory, lived situation, or first-person narrative.

Positive indicators:
- `mi e successo`
- `nel mio caso`
- `ci sono passata`
- developed first-person account

### 9. `advisory`

Definition:
advice, suggestion, or recommendation about what to do.

Positive indicators:
- `ti consiglio`
- `prova a`
- `dovresti`
- `leggi questo`
- `guarda questo`

Do not use if:
- the comment mainly tags another user to show them the post -> `sharing`

### 10. `situated_experts`

Definition:
comment organized by a declared role or expertise position.

Positive indicators:
- `da psicoterapeuta`
- `sono una psicologa`
- `da mamma`
- `da medico`

### 11. `defensive`

Definition:
defense of Cami, the post, or discussion boundaries.

Positive indicators:
- `lasciatela stare`
- `avete rotto`
- `non capisco certi commenti`
- boundary-setting against critics

### 12. `sharing`

Definition:
the comment is mainly used to involve another user, tag a friend, recommend the post, or forward the content in public.

Positive indicators:
- `@nome`
- `@nome guarda`
- `@nome leggi`
- `questa sei tu @nome`
- comment whose main act is "showing this post to someone"

Do not use if:
- there is no explicit addressee and no real share-to-someone move
- the mention is just addressing someone inside a reply thread
- there is strong insider familiarity -> `social_circle`

### 13. `social_circle`

Definition:
insider or long-term familiarity with Cami, shared lore, recurring traits, backstage references, or close-circle authorship.

Positive indicators:
- known creator/friend-circle usernames
- `septum storto`
- references to recurring details strongly tied to Cami
- repeated-lore jokes
- comments that presuppose longstanding familiarity

Important note:
a follower does not need to be a real-life friend to count here. Clear recurring-familiarity from the fan side can also qualify.

### 14. `other_unclear`

Definition:
still too ambiguous or too generic after using the available context.

Use only if:
- the function remains genuinely unclear
- the comment is too minimal to resolve
- the caption still does not clarify the communicative act

## Key disambiguation rules

### `affective` vs `aesthetic`

- choose `affective` for warmth, admiration of Cami, closeness
- choose `aesthetic` for beauty, attractiveness, styling, or visual admiration
- if the compliment is brief and the visual target is not clear, default to `affective`
- if words like `fantastica`, `bona`, `figa`, or `ciaone` are paired with heart-eyes emojis, choose `affective`
- if the comment is only heart-eyes or similar "in love" emojis, choose `affective`

### `affective` vs `validating`

- choose `affective` when the praise is about Cami
- choose `validating` when the support is for a point or message

### `humorous` vs `other_unclear`

- if the caption makes the joke readable, choose `humorous`
- do not require explicit laughter markers

### `sharing` vs `social_circle`

- choose `sharing` for tag-and-show or tag-and-recommend behavior
- do not use `sharing` without an explicit addressee or clear share-to-friend move
- choose `social_circle` only for real insider evidence or clear recurring familiarity

### `sharing` vs `advisory`

- choose `sharing` for `@nome guarda`
- choose `advisory` when the advice is addressed to Cami or the audience rather than used to tag someone in

## Implications for prompt design

The prompt should explicitly tell the LLM that:

- caption matters for short contextual comments
- generic praise is not automatically `other_unclear`
- bare top-level mentions are often `sharing`
- recurring familiarity with Cami can count as `social_circle`
- exactly one primary label must be returned
