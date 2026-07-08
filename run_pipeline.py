#!/usr/bin/env python3
"""Clipsmith pipeline orchestrator.

Usage:
    python run_pipeline.py <youtube-url> [--clips 5] [--workdir runs/ep1]
                            [--max-minutes N] [--dry-run]
"""

import argparse
import json
import os
import re
import subprocess
import time

VIDEO_CROP_FILTER = "crop=ih*9/16:ih,scale=1080:1920"
SENTENCE_PAD = 0.3
MIN_CLIP_LEN = 20
MAX_CLIP_LEN = 75
GROQ_MODEL = "llama-3.3-70b-versatile"
CONTEXT_WINDOW = 15.0
MAX_REFINEMENT = 10.0


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def default_workdir(url: str) -> str:
    try:
        result = subprocess.run(
            ["yt-dlp", "--no-playlist", "--get-id", "--", url],
            check=True, capture_output=True, text=True,
        )
        video_id = result.stdout.strip().splitlines()[-1]
    except Exception:
        video_id = "episode"
    return os.path.join("runs", video_id)


def _extract_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _call_llm_json(client, system: str, user_prompt: str):
    for attempt in range(2):
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content
        try:
            return json.loads(_extract_json(raw))
        except json.JSONDecodeError:
            if attempt == 1:
                raise
            user_prompt = (
                user_prompt
                + "\n\nYour previous response was not valid JSON. Return"
                " ONLY the JSON, no prose, no markdown fences."
            )


def _sentences_in_range(transcript: dict, start: float, end: float) -> str:
    return " ".join(
        s["text"] for s in transcript["sentences"] if s["end"] > start and s["start"] < end
    )


def _validate_candidates(raw, duration: float) -> list[dict]:
    """Coerce pass-1 output into clean {start, end, summary} dicts,
    dropping anything malformed so one bad element can't crash pass 2."""
    if not isinstance(raw, list):
        raise ValueError(f"expected a JSON array of candidates, got {type(raw).__name__}")
    valid = []
    for c in raw:
        try:
            start, end = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= duration + 1):
            continue
        valid.append({"start": start, "end": end, "summary": str(c.get("summary", ""))})
    if not valid:
        raise ValueError("no usable candidates in pass-1 response")
    if len(valid) < len(raw):
        log(f"    dropped {len(raw) - len(valid)} malformed candidate(s)")
    return valid


# --------------------------------------------------------------------------
# Stage 1: ingest
# --------------------------------------------------------------------------

def ingest(url: str, workdir: str, max_minutes: float | None) -> tuple[str, str]:
    video_path = os.path.join(workdir, "episode.mp4")
    audio_path = os.path.join(workdir, "episode.wav")

    if os.path.exists(video_path) and os.path.exists(audio_path):
        log("[cache] episode.mp4 / episode.wav already exist, skipping download")
        return video_path, audio_path

    os.makedirs(workdir, exist_ok=True)
    log(f"==> Downloading {url}")
    raw_path = os.path.join(workdir, "episode_raw.mp4")
    run([
        "yt-dlp", "--no-playlist",
        # Best <=1080p video + audio, merged by ffmpeg — the pre-merged
        # "-f mp4" fallback alone is usually only 360p, which upscales
        # badly to 1080x1920.
        "-f", "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]",
        "--merge-output-format", "mp4",
        "-o", raw_path,
        "--", url,
    ])

    if max_minutes:
        log(f"==> Trimming to first {max_minutes} minutes")
        run(["ffmpeg", "-y", "-i", raw_path, "-t", str(max_minutes * 60), "-c", "copy", video_path])
        os.remove(raw_path)
    else:
        os.rename(raw_path, video_path)

    log("==> Extracting audio (16kHz mono wav for transcription)")
    run(["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", audio_path])

    return video_path, audio_path


# --------------------------------------------------------------------------
# Stage 2: transcribe
# --------------------------------------------------------------------------

def transcribe(audio_path: str, workdir: str) -> dict:
    transcript_path = os.path.join(workdir, "transcript.json")
    if os.path.exists(transcript_path):
        log("[cache] transcript.json already exists, skipping transcription")
        with open(transcript_path) as f:
            return json.load(f)

    from faster_whisper import WhisperModel

    device, compute_type, model_size = "cpu", "int8", "medium"
    try:
        import torch
        if torch.cuda.is_available():
            device, compute_type, model_size = "cuda", "float16", "large-v3"
    except ImportError:
        pass

    log(f"==> Transcribing audio (faster-whisper: {model_size}, device={device})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    log(f"    Audio duration: {info.duration / 60:.1f} min — estimating ETA as we go")

    start_time = time.time()
    sentences, words = [], []
    for seg in segments:
        sentences.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        for w in seg.words or []:
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

        elapsed = time.time() - start_time
        progress = seg.end / info.duration if info.duration else 0
        eta_str = ""
        if 0 < progress < 1:
            eta_min = (elapsed / progress - elapsed) / 60
            eta_str = f"  (ETA {eta_min:.1f}min)"
        log(f"    [{seg.start:6.1f}s]{eta_str} {seg.text.strip()[:70]}")

    log(f"    Transcription finished in {(time.time() - start_time) / 60:.1f} min")

    transcript = {"duration": info.duration, "sentences": sentences, "words": words}
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=2)

    return transcript


# --------------------------------------------------------------------------
# Stage 3: find moments (LLM pass 1)
# --------------------------------------------------------------------------

def find_moments(transcript: dict, dry_run: bool, workdir: str, num_candidates: int = 15) -> list[dict]:
    candidates_path = os.path.join(workdir, "candidates.json")
    if os.path.exists(candidates_path):
        log("[cache] candidates.json already exists, skipping moment-finding")
        with open(candidates_path) as f:
            return json.load(f)

    if dry_run:
        log("==> Finding viral moments (--dry-run: using 3 fake candidates)")
        duration = transcript["duration"]
        fake_specs = [
            (0.12, "A bold contrarian claim early in the episode"),
            (0.45, "A concrete story with a specific tactical takeaway"),
            (0.75, "A punchy, quotable closing thought"),
        ]
        candidates = []
        for frac, summary in fake_specs:
            start = max(0.0, frac * duration)
            end = min(duration, start + 35.0)
            if end - start >= MIN_CLIP_LEN:
                candidates.append({"start": start, "end": end, "summary": summary})
    else:
        import prompts
        from groq import Groq

        log("==> Finding viral moments (Groq pass 1)")
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        transcript_text = "\n".join(
            f"[{s['start']:.1f}s] {s['text']}" for s in transcript["sentences"]
        )
        prompt = prompts.build_find_moments_prompt(transcript_text, num_candidates)
        candidates = _validate_candidates(
            _call_llm_json(client, prompts.FIND_MOMENTS_SYSTEM, prompt),
            transcript["duration"],
        )
        log(f"    Groq found {len(candidates)} candidate moments")

    with open(candidates_path, "w") as f:
        json.dump(candidates, f, indent=2)

    return candidates


# --------------------------------------------------------------------------
# Stage 4: score clips (LLM pass 2)
# --------------------------------------------------------------------------

FAKE_METADATA = [
    {
        "score": 9,
        "title": "This one habit changed everything",
        "hook_line": "Nobody tells you this until it's too late.",
        "description": "A candid, contrarian take you won't hear anywhere else.",
        "hashtags": ["#business", "#mindset", "#entrepreneur"],
        "reason": "Strong contrarian hook, self-contained, specific claim.",
    },
    {
        "score": 8,
        "title": "The exact tactic that doubled our results",
        "hook_line": "Here's exactly what we did, step by step.",
        "description": "A concrete, replicable tactic pulled straight from the episode.",
        "hashtags": ["#growth", "#tactics", "#smallbusiness"],
        "reason": "Concrete and actionable, clear before/after framing.",
    },
    {
        "score": 7,
        "title": "The line that says it all",
        "hook_line": "If you remember one thing, remember this.",
        "description": "A punchy, quotable closer that works as a standalone thought.",
        "hashtags": ["#quote", "#wisdom", "#motivation"],
        "reason": "Quotable and short, but slightly weaker opening hook.",
    },
]


def score_clips(candidates: list[dict], transcript: dict, dry_run: bool, workdir: str) -> list[dict]:
    scored_path = os.path.join(workdir, "scored_clips.json")
    if os.path.exists(scored_path):
        log("[cache] scored_clips.json already exists, skipping scoring")
        with open(scored_path) as f:
            return json.load(f)

    scored = []

    if dry_run:
        log("==> Scoring clips (--dry-run: using fake scores)")
        for i, cand in enumerate(candidates):
            meta = FAKE_METADATA[i % len(FAKE_METADATA)]
            clip = {**cand, **meta}
            scored.append(clip)
            log(f'    [{i + 1}/{len(candidates)}] score={clip["score"]}  "{clip["title"]}"  — {clip["reason"]}')
    else:
        import prompts
        from groq import Groq

        log("==> Scoring clips (Groq pass 2)")
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        duration = transcript["duration"]
        for i, cand in enumerate(candidates):
            start, end = cand["start"], cand["end"]
            clip_text = _sentences_in_range(transcript, start, end)
            context_before = _sentences_in_range(transcript, max(0.0, start - CONTEXT_WINDOW), start)
            context_after = _sentences_in_range(transcript, end, min(duration, end + CONTEXT_WINDOW))
            prompt = prompts.build_score_clip_prompt(start, end, clip_text, context_before, context_after)
            result = _call_llm_json(client, prompts.SCORE_CLIP_SYSTEM, prompt)

            # Guard against the LLM returning snippet-relative (not
            # absolute) timestamps by discarding refinements that drift
            # too far from the original candidate window.
            if abs(result.get("start", start) - start) > MAX_REFINEMENT:
                result["start"] = start
            if abs(result.get("end", end) - end) > MAX_REFINEMENT:
                result["end"] = end

            # Defaults for any field pass 2 forgot, so metadata assembly
            # at the end of the run can't KeyError.
            try:
                result["score"] = int(result.get("score", 5))
            except (TypeError, ValueError):
                result["score"] = 5
            result.setdefault("title", cand.get("summary", "Untitled clip")[:60])
            result.setdefault("hook_line", "")
            result.setdefault("description", cand.get("summary", ""))
            result.setdefault("hashtags", [])
            result.setdefault("reason", "")

            clip = {**cand, **result}
            scored.append(clip)
            log(f'    [{i + 1}/{len(candidates)}] score={clip["score"]}  "{clip["title"]}"  — {clip["reason"]}')

    scored.sort(key=lambda c: c["score"], reverse=True)

    with open(scored_path, "w") as f:
        json.dump(scored, f, indent=2)

    return scored


def select_top_clips(scored: list[dict], n: int) -> list[dict]:
    """Take the highest-scored clips, skipping any that substantially
    overlap an already-selected one so the final set covers N distinct
    moments (pass 1 sometimes proposes near-duplicate windows)."""
    selected = []
    for cand in scored:
        if len(selected) >= n:
            break
        dup = False
        for s in selected:
            overlap = min(cand["end"], s["end"]) - max(cand["start"], s["start"])
            shorter = min(cand["end"] - cand["start"], s["end"] - s["start"])
            if shorter > 0 and overlap > 0.5 * shorter:
                dup = True
                break
        if dup:
            log(f'    [skip] "{cand.get("title", "?")}" overlaps an already-selected clip')
        else:
            selected.append(cand)
    return selected


# --------------------------------------------------------------------------
# Boundary snapping
# --------------------------------------------------------------------------

def snap_boundaries(clips: list[dict], transcript: dict) -> list[dict]:
    sentences = transcript["sentences"]
    duration = transcript["duration"]
    starts = [s["start"] for s in sentences]
    ends = [s["end"] for s in sentences]

    snapped = []
    for clip in clips:
        start = (min(starts, key=lambda c: abs(c - clip["start"])) if starts else clip["start"]) - SENTENCE_PAD
        end = (min(ends, key=lambda c: abs(c - clip["end"])) if ends else clip["end"]) + SENTENCE_PAD
        start = max(0.0, start)
        end = min(duration, end)
        if end - start < MIN_CLIP_LEN:
            end = min(duration, start + MIN_CLIP_LEN)
        if end - start > MAX_CLIP_LEN:
            end = start + MAX_CLIP_LEN
        snapped.append({**clip, "start": round(start, 2), "end": round(end, 2)})

    return snapped


# --------------------------------------------------------------------------
# Cut + crop, caption burn
# --------------------------------------------------------------------------

def cut_and_crop(video_path: str, clip: dict, out_path: str) -> str:
    if os.path.exists(out_path):
        log(f"[cache] {os.path.basename(out_path)} already cut, skipping")
        return out_path

    crop_filter = VIDEO_CROP_FILTER
    try:
        import facecrop

        face = facecrop.compute_crop_x(video_path, clip["start"], clip["end"])
    except Exception as e:
        log(f"        face detection unavailable ({e}), using center crop")
        face = None

    if face:
        crop_x, crop_w, src_h = face
        crop_filter = f"crop={crop_w}:{src_h}:{crop_x}:0,scale=1080:1920"
        log(f"        face-aware crop: window x={crop_x} (of {crop_w}px wide)")
    else:
        log("        no face detected, using center crop")

    run([
        "ffmpeg", "-y",
        "-ss", str(clip["start"]),
        "-i", video_path,
        "-t", str(clip["end"] - clip["start"]),
        "-vf", crop_filter,
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-c:a", "aac",
        out_path,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out_path


def burn_captions(cropped_path: str, clip: dict, transcript: dict, out_path: str) -> str:
    import captions as captions_mod

    if os.path.exists(out_path):
        log(f"[cache] {os.path.basename(out_path)} already captioned, skipping")
        return out_path

    clip_words = [
        {"word": w["word"], "start": w["start"] - clip["start"], "end": w["end"] - clip["start"]}
        for w in transcript["words"]
        if clip["start"] <= w["start"] < clip["end"]
    ]

    ass_path = out_path.replace(".mp4", ".ass")
    captions_mod.build_ass_file(clip_words, ass_path)

    escaped_ass = ass_path.replace(":", "\\:")
    run([
        "ffmpeg", "-y", "-i", cropped_path,
        "-vf", f"ass={escaped_ass}",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-c:a", "copy",
        out_path,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out_path


# --------------------------------------------------------------------------
# Stage 5: newsletter (LLM pass 3)
# --------------------------------------------------------------------------

def generate_newsletter(transcript: dict, dry_run: bool, outputs_dir: str) -> str:
    newsletter_path = os.path.join(outputs_dir, "newsletter.md")
    if os.path.exists(newsletter_path):
        log("[cache] newsletter.md already exists, skipping newsletter generation")
        return newsletter_path

    if dry_run:
        log("==> Generating newsletter (--dry-run: using placeholder)")
        content = (
            "# [DRY RUN] Placeholder Subject Line\n\n"
            "This is a placeholder newsletter generated in --dry-run mode.\n"
        )
    else:
        import prompts
        from groq import Groq

        log("==> Generating newsletter (Groq pass 3)")
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        full_text = "\n".join(
            f"[{s['start']:.1f}s] {s['text']}" for s in transcript["sentences"]
        )
        prompt = prompts.build_newsletter_prompt(full_text)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": prompts.NEWSLETTER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()

    with open(newsletter_path, "w") as f:
        f.write(content)

    log(f"    Newsletter written to {newsletter_path}")
    return newsletter_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clipsmith: long-form episode -> short-form clips + newsletter")
    parser.add_argument("url", help="YouTube URL of the source episode")
    parser.add_argument("--clips", type=int, default=5, help="Number of clips to produce")
    parser.add_argument("--workdir", default=None, help="Working directory for this episode (default: runs/<video-id>)")
    parser.add_argument("--max-minutes", type=float, default=None, help="Trim input to first N minutes (cheap iteration)")
    parser.add_argument("--dry-run", action="store_true", help="Skip Groq calls, use fake candidates")
    args = parser.parse_args()

    if not args.url.startswith(("http://", "https://")):
        parser.error("url must start with http:// or https://")

    workdir = args.workdir or default_workdir(args.url)
    outputs_dir = os.path.join(workdir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    log(f"Clipsmith — workdir: {workdir}")
    log("=" * 60)

    video_path, audio_path = ingest(args.url, workdir, args.max_minutes)
    transcript = transcribe(audio_path, workdir)
    candidates = find_moments(transcript, args.dry_run, workdir)
    scored = score_clips(candidates, transcript, args.dry_run, workdir)

    top_clips = select_top_clips(scored, args.clips)

    log(f"==> Snapping {len(top_clips)} cut boundaries to sentence edges")
    top_clips = snap_boundaries(top_clips, transcript)

    log("==> Cutting, cropping, and captioning clips")
    metadata = []
    for i, clip in enumerate(top_clips, start=1):
        name = f"short_{i:02d}"
        cropped_path = os.path.join(outputs_dir, f"{name}_cropped.mp4")
        final_path = os.path.join(outputs_dir, f"{name}.mp4")

        log(f'    [{i}/{len(top_clips)}] {name}: {clip["start"]:.1f}s -> {clip["end"]:.1f}s  "{clip["title"]}"')
        cut_and_crop(video_path, clip, cropped_path)
        burn_captions(cropped_path, clip, transcript, final_path)

        metadata.append({
            "file": os.path.basename(final_path),
            "score": clip["score"],
            "start": clip["start"],
            "end": clip["end"],
            "title": clip["title"],
            "hook_line": clip["hook_line"],
            "description": clip["description"],
            "hashtags": clip["hashtags"],
            "reason": clip["reason"],
        })

    metadata_path = os.path.join(outputs_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    generate_newsletter(transcript, args.dry_run, outputs_dir)

    log("=" * 60)
    log(f"Done. {len(metadata)} clips written to {outputs_dir}/")


if __name__ == "__main__":
    main()
