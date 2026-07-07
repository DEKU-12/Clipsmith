"""LLM prompt templates for Clipsmith (used with the Groq API).

Wired up in M2 — see run_pipeline.py's find_moments()/score_clips()/
generate_newsletter(). All prompts instruct the model to return pure
JSON; callers strip ``` fences defensively before json.loads.
"""

FIND_MOMENTS_SYSTEM = (
    "You are an expert short-form video producer who has cut thousands of "
    "viral clips from long-form podcasts and livestreams. You have a sharp "
    "eye for moments that stop the scroll: strong hooks, contrarian takes, "
    "specific tactical advice, emotional stories, and surprising reveals."
)


def build_find_moments_prompt(transcript_text: str, num_candidates: int = 15) -> str:
    return f"""Below is a full timestamped transcript of a long-form episode.
Find the {num_candidates} best moments that could become standalone
short-form vertical clips (20-75 seconds each).

Prioritize moments that:
- Open with a strong hook in the first 5 seconds (a bold claim, a question,
  a number, or a contrarian statement)
- Are self-contained: a viewer with zero prior context understands and
  cares within the first sentence
- Contain a specific, concrete idea (a tactic, a story, a number) rather
  than vague commentary
- Have a natural beginning and end (don't cut off mid-thought)

Return ONLY a JSON array, no prose, no markdown fences. Each element:
{{"start": <float seconds>, "end": <float seconds>, "summary": "<one sentence on what this moment is about>"}}

Transcript:
{transcript_text}
"""


SCORE_CLIP_SYSTEM = (
    "You are an expert short-form video editor grading candidate clips for "
    "how likely they are to perform well on TikTok/Reels/Shorts. Hook "
    "strength matters more than any other factor."
)


def build_score_clip_prompt(
    candidate_start: float, candidate_end: float, clip_text: str, context_before: str, context_after: str
) -> str:
    return f"""Candidate clip transcript (with ~15s of context on each side
for judgment only — do not include the context in the clip itself):

The candidate clip currently spans {candidate_start:.1f}s to {candidate_end:.1f}s,
in absolute seconds measured from the start of the full episode.

--- CONTEXT BEFORE ---
{context_before}

--- CANDIDATE CLIP ({candidate_start:.1f}s to {candidate_end:.1f}s) ---
{clip_text}

--- CONTEXT AFTER ---
{context_after}

Score this candidate 1-10 for short-form virality. Hook strength (does the
first 5 seconds grab attention?) is the heaviest weighted factor. The clip
must be fully self-contained with zero prior context required.

You may refine the start/end times slightly (at most a few seconds each
way) to tighten the hook or land a cleaner ending, but keep the clip
between 20 and 75 seconds. Your refined start/end MUST stay in absolute
episode seconds, close to the given {candidate_start:.1f}s-{candidate_end:.1f}s
range — do NOT reset them to be relative to the clip snippet (e.g. do not
return a start near 0 unless {candidate_start:.1f}s is actually near 0).

Return ONLY a JSON object, no prose, no markdown fences:
{{
  "start": <float seconds, absolute episode time, near {candidate_start:.1f}>,
  "end": <float seconds, absolute episode time, near {candidate_end:.1f}>,
  "score": <int 1-10>,
  "title": "<punchy title, max 60 chars>",
  "hook_line": "<the exact opening line/moment that hooks viewers>",
  "description": "<1-2 sentence caption for the post>",
  "hashtags": ["<tag1>", "<tag2>", "..."],
  "reason": "<one sentence on why this scores what it does>"
}}
"""


NEWSLETTER_SYSTEM = (
    "You are a newsletter editor who writes tight, punchy recap emails for "
    "a Beehiiv newsletter based on long-form podcast/livestream episodes."
)


def build_newsletter_prompt(full_transcript: str) -> str:
    return f"""Write a Beehiiv-ready newsletter draft recapping this episode.

Structure (use Markdown):
1. Subject line (compelling, under 60 chars)
2. A 2-3 sentence TL;DR
3. 3-4 key takeaways, each with a short header and 2-3 sentences
4. 1-2 direct quotes pulled verbatim from the transcript
5. A closing CTA (e.g. watch the full episode / subscribe)

Only use information present in the transcript. Do not invent topics,
numbers, or quotes that aren't in the source material.

Transcript:
{full_transcript}
"""
