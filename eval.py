#!/usr/bin/env python3
"""Evaluate a finished Clipsmith run for hallucination and spec violations.

Usage:
    python eval.py runs/<name> [--no-llm]

Layer 1 (deterministic, free):
  - every clip 20-75s, inside the episode, metadata complete, files exist
  - no two selected clips substantially overlap
  - each clip's hook_line fuzzy-matches the transcript in its time range
  - every blockquote in newsletter.md fuzzy-matches the transcript

Layer 2 (LLM-as-judge via Groq, skipped with --no-llm or no GROQ_API_KEY):
  - clip titles/descriptions: every claim supported by the clip transcript
  - newsletter body: no topics/claims absent from the episode

Exit code 0 = all checks passed, 1 = at least one failure.
Meant for real runs — dry-run metadata is fake and will (correctly) fail.
"""

import argparse
import difflib
import json
import os
import re
import sys

FUZZY_THRESHOLD = 0.8
MIN_CLIP_LEN = 20
MAX_CLIP_LEN = 75
HOOK_WINDOW_PAD = 10.0

passed, failed = 0, 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global passed, failed
    mark = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f"  ({detail})" if detail and not ok else ""
    print(f"  [{mark}] {label}{suffix}")


def _normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s']", " ", text.lower()).split()


def fuzzy_contains(needle: str, haystack: str) -> float:
    """Best similarity of `needle` against any same-length word window of
    `haystack`. 1.0 = verbatim (modulo case/punctuation)."""
    n_words, h_words = _normalize(needle), _normalize(haystack)
    if not n_words or not h_words:
        return 0.0
    n_str = " ".join(n_words)
    window = len(n_words)
    best = 0.0
    for i in range(max(1, len(h_words) - window + 1)):
        w_str = " ".join(h_words[i : i + window])
        best = max(best, difflib.SequenceMatcher(None, n_str, w_str).ratio())
        if best > 0.99:
            break
    return best


def quote_grounded(quote: str, transcript_text: str) -> float:
    """Quotes may compress with '...' — every fragment must match."""
    fragments = [f for f in re.split(r"\.{3}|…", quote) if len(_normalize(f)) >= 3]
    if not fragments:
        fragments = [quote]
    return min(fuzzy_contains(f, transcript_text) for f in fragments)


def eval_clips(metadata: list[dict], transcript: dict, outputs_dir: str) -> None:
    duration = transcript["duration"]
    required = ["file", "score", "start", "end", "title", "hook_line", "description", "hashtags", "reason"]

    for i, clip in enumerate(metadata, start=1):
        print(f'\nClip {i}: "{clip.get("title", "?")}"')

        missing = [k for k in required if k not in clip or clip[k] in (None, "")]
        check(not missing, "metadata complete", f"missing: {missing}")

        start, end = clip.get("start", 0), clip.get("end", 0)
        length = end - start
        # 0.1s epsilon: metadata timestamps are rounded to 2 decimals.
        check(
            MIN_CLIP_LEN - 0.1 <= length <= MAX_CLIP_LEN + 0.1,
            f"duration {length:.1f}s within 20-75s",
        )
        check(0 <= start < end <= duration + 1, "timestamps inside episode")
        check(os.path.exists(os.path.join(outputs_dir, clip.get("file", ""))), "clip file exists")

        hook = clip.get("hook_line", "")
        if hook:
            window_text = " ".join(
                s["text"] for s in transcript["sentences"]
                if s["end"] > start - HOOK_WINDOW_PAD and s["start"] < end + HOOK_WINDOW_PAD
            )
            score = fuzzy_contains(hook, window_text)
            check(
                score >= FUZZY_THRESHOLD,
                f"hook_line grounded in clip transcript (match={score:.2f})",
                f'hook not found near {start:.0f}s-{end:.0f}s: "{hook[:60]}"',
            )

    for a in range(len(metadata)):
        for b in range(a + 1, len(metadata)):
            ca, cb = metadata[a], metadata[b]
            overlap = min(ca["end"], cb["end"]) - max(ca["start"], cb["start"])
            shorter = min(ca["end"] - ca["start"], cb["end"] - cb["start"])
            check(
                not (shorter > 0 and overlap > 0.5 * shorter),
                f"clips {a + 1} and {b + 1} are distinct moments",
                f"{overlap:.1f}s overlap",
            )


def eval_newsletter(newsletter: str, transcript_text: str) -> None:
    print("\nNewsletter:")
    quotes = [m.strip(" >\"'") for m in re.findall(r"^>\s*(.+)$", newsletter, re.MULTILINE)]
    if not quotes:
        print("  [note] no blockquotes found to verify")
    for q in quotes:
        score = quote_grounded(q, transcript_text)
        check(
            score >= FUZZY_THRESHOLD,
            f"quote verbatim in transcript (match={score:.2f})",
            f'"{q[:60]}..."',
        )


def llm_judge(metadata: list[dict], newsletter: str, transcript: dict) -> None:
    from groq import Groq

    import run_pipeline as rp

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    system = (
        "You are a strict fact-checker. Judge ONLY whether the generated "
        "text is supported by the source transcript. Respond with pure "
        'JSON: {"supported": true/false, "unsupported_claims": ["..."]}'
    )

    print("\nLLM judge — clip metadata:")
    for i, clip in enumerate(metadata, start=1):
        span = rp._sentences_in_range(transcript, clip["start"], clip["end"])
        prompt = (
            f"SOURCE TRANSCRIPT (clip {clip['start']:.0f}s-{clip['end']:.0f}s):\n{span}\n\n"
            f"GENERATED TITLE: {clip['title']}\n"
            f"GENERATED DESCRIPTION: {clip['description']}\n\n"
            "Is every factual claim in the title and description supported "
            "by the source transcript? Style/tone is fine; invented facts are not."
        )
        verdict = rp._call_llm_json(client, system, prompt, temperature=0.0)
        check(
            bool(verdict.get("supported")),
            f'clip {i} title/description supported',
            "; ".join(verdict.get("unsupported_claims", []))[:120],
        )

    print("\nLLM judge — newsletter:")
    full_text = " ".join(s["text"] for s in transcript["sentences"])
    prompt = (
        f"SOURCE TRANSCRIPT:\n{full_text}\n\n"
        f"GENERATED NEWSLETTER:\n{newsletter}\n\n"
        "Does the newsletter contain any topics, facts, or numbers that "
        "are NOT present in the source transcript? Direct blockquotes have "
        "already been verified verbatim separately — ignore them. Also "
        "ignore calls-to-action, subscribe/watch prompts, links, subject "
        "lines, and formatting boilerplate: those are intentional newsletter "
        "structure, not factual claims. Judge only whether the prose "
        "misrepresents or invents episode content."
    )
    verdict = rp._call_llm_json(client, system, prompt, temperature=0.0)
    check(
        bool(verdict.get("supported")),
        "newsletter claims supported",
        "; ".join(verdict.get("unsupported_claims", []))[:200],
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Clipsmith run for hallucination")
    parser.add_argument("workdir", help="Run directory, e.g. runs/yHW_seFM3nA")
    parser.add_argument("--no-llm", action="store_true", help="Skip the LLM-as-judge layer")
    args = parser.parse_args()

    outputs_dir = os.path.join(args.workdir, "outputs")
    with open(os.path.join(args.workdir, "transcript.json")) as f:
        transcript = json.load(f)
    with open(os.path.join(outputs_dir, "metadata.json")) as f:
        metadata = json.load(f)
    newsletter = ""
    newsletter_path = os.path.join(outputs_dir, "newsletter.md")
    if os.path.exists(newsletter_path):
        with open(newsletter_path) as f:
            newsletter = f.read()

    print(f"Evaluating {args.workdir} — {len(metadata)} clips")
    print("=" * 60)

    eval_clips(metadata, transcript, outputs_dir)
    transcript_text = " ".join(s["text"] for s in transcript["sentences"])
    if newsletter:
        eval_newsletter(newsletter, transcript_text)

    if args.no_llm or not os.environ.get("GROQ_API_KEY"):
        print("\n[skipped] LLM-as-judge layer (--no-llm or GROQ_API_KEY unset)")
    else:
        llm_judge(metadata, newsletter, transcript)

    print("=" * 60)
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
