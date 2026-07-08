# Clipsmith

Turns a long-form YouTube livestream/interview into (a) captioned 9:16
short-form clips with publish-ready metadata, and (b) a newsletter draft —
from a single command.

Ingest → transcribe → find viral moments → score & title each candidate →
snap cuts to sentence boundaries → crop to 9:16 → burn karaoke captions →
draft a newsletter. Every stage caches its output to disk, so re-running
after a crash or a tweak only recomputes what's missing.

## Architecture

```
clipsmith/
├── run_pipeline.py      # CLI + orchestrator
├── prompts.py           # LLM prompt templates (moment-finding, scoring, newsletter)
├── captions.py          # Word-timestamps -> styled .ass karaoke subtitle file
├── requirements.txt
└── runs/<name>/         # per-episode working dir (gitignored)
    ├── episode.mp4, episode.wav
    ├── transcript.json          # {"sentences": [...], "words": [...]} with start/end times
    ├── candidates.json          # pass 1 output (12-16 candidate moments)
    ├── scored_clips.json        # pass 2 output, sorted by score
    └── outputs/
        ├── short_01.mp4 ...     # final captioned 9:16 clips
        ├── metadata.json        # title/hook/description/hashtags/score per clip
        └── newsletter.md
```

**Pipeline stages:**

```
 yt-dlp          faster-whisper       LLM pass 1          LLM pass 2
┌────────┐      ┌─────────────┐     ┌───────────┐      ┌──────────────┐
│ ingest │ ───▶ │ transcribe  │ ──▶ │find moments│ ──▶ │ score + title │
└────────┘      │(word-level  │     │(12-16 cands)│     │  each clip   │
                 │ timestamps)│     └───────────┘      └──────┬───────┘
                 └─────────────┘                              │
                                                                ▼
 ┌──────────────┐      ┌────────────┐      ┌──────────────────────┐
 │ burn karaoke │ ◀─── │ ffmpeg cut │ ◀─── │ snap boundaries to    │
 │  captions    │      │ + crop 9:16│      │ sentence starts/ends  │
 └──────┬───────┘      └────────────┘      └──────────────────────┘
        │
        ▼
 outputs/short_NN.mp4 + metadata.json + newsletter.md
```

Model: **Groq's `llama-3.3-70b-versatile`** (free tier) handles all three
LLM passes — moment-finding, per-clip scoring, and the newsletter draft.
It's swapped in for a paid LLM API so this demo runs end-to-end at zero
API cost and can be re-run freely while iterating.

## Setup

1. **Python 3.10+**

2. **ffmpeg with libass support.** The default Homebrew `ffmpeg` formula
   does *not* include libass, which is required to burn `.ass` karaoke
   captions. Install the full build instead:
   ```bash
   brew uninstall ffmpeg        # if already installed without libass
   brew install homebrew-ffmpeg/ffmpeg/ffmpeg
   ```
   Verify: `ffmpeg -filters | grep ass` should list the `ass` filter.

3. **yt-dlp** on PATH:
   ```bash
   brew install yt-dlp
   ```

4. **Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Groq API key** (free, no credit card required):
   - Get one at [console.groq.com/keys](https://console.groq.com/keys)
   - Add it to your shell profile:
     ```bash
     echo 'export GROQ_API_KEY=your-key-here' >> ~/.zshrc
     source ~/.zshrc
     ```

## Usage

```bash
python run_pipeline.py <youtube-url> [--clips 5] [--workdir runs/ep1] [--max-minutes N] [--dry-run]
```

- `--clips N` — number of clips to produce (default 5)
- `--workdir DIR` — working directory for this episode (default `runs/<video-id>`)
- `--max-minutes N` — trim to the first N minutes before transcription, for cheap/fast iteration
- `--dry-run` — skip all LLM calls, use 3 hardcoded fake candidates so the cut/caption stages can be tested at zero cost

**Cheap iteration example** (no LLM calls, ~30s):
```bash
python run_pipeline.py "https://youtube.com/watch?v=..." --dry-run --max-minutes 3 --clips 2
```

**Full run:**
```bash
python run_pipeline.py "https://youtube.com/watch?v=..." --clips 5
```

## Optional: browser UI

`app.py` is a small Streamlit front-end over the same pipeline — paste a
URL, set options, watch the console log stream live, then browse the
resulting clips (with video players), metadata, and newsletter draft in
the browser instead of the terminal. It shells out to `run_pipeline.py`
under the hood, so the CLI stays the source of truth.

```bash
streamlit run app.py
```

## Sample console output

```
Clipsmith — workdir: runs/yHW_seFM3nA
============================================================
[cache] episode.mp4 / episode.wav already exist, skipping download
==> Transcribing audio (faster-whisper: medium, device=cpu)
    Audio duration: 21.7 min — estimating ETA as we go
    [ 683.3s]  (ETA 4.2min) You have to use the cards that you're dealt, right?
    ...
    Transcription finished in 9.0 min
==> Finding viral moments (Groq pass 1)
    Groq found 16 candidate moments
==> Scoring clips (Groq pass 2)
    [1/16] score=8  "Microplastics Inside?"  — The clip scores high due to its alarming
                                                and thought-provoking hook, immediately
                                                grabbing attention.
    [2/16] score=8  "Poisoning Ourselves?"  — Thought-provoking, alarming hook that
                                               immediately grabs attention.
    ...
==> Snapping 5 cut boundaries to sentence edges
==> Cutting, cropping, and captioning clips
    [1/5] short_01: 18.6s -> 38.6s  "Microplastics Inside?"
    ...
==> Generating newsletter (Groq pass 3)
    Newsletter written to runs/yHW_seFM3nA/outputs/newsletter.md
============================================================
Done. 5 clips written to runs/yHW_seFM3nA/outputs/
```

## How the demo video was recorded

1. Ran the full pipeline against a real ~20-minute interview episode with
   no `--max-minutes` trim, so the console output shows genuine
   transcription progress, ETA estimates, and real LLM scoring reasoning
   — not a canned/fake run.
2. Screen-recorded the terminal (QuickTime Player → File → New Screen
   Recording) from the command invocation through to the final
   `Done. N clips written to ...` line, so the pacing and per-clip
   scoring printout are unedited.
3. Cut to the Finder/QuickTime preview of 2-3 output clips to show the
   captioned 9:16 result, karaoke word-highlighting, and clean cut
   boundaries (no mid-word cuts).
4. Opened `metadata.json` and `newsletter.md` briefly to show the
   structured publish-ready output alongside the video clips.

## Notes

- **The 9:16 crop is face-aware.** Each clip samples a handful of frames,
  runs OpenCV's YuNet face detector (`facecrop.py`, model vendored in the
  repo), and slides the crop window to center on the dominant face instead
  of blindly cropping the frame middle — so side-by-side podcast layouts
  crop to a person, not the wall between them. Falls back to a center crop
  when no face is found. True active-speaker tracking (following whoever
  is talking) is a known next step.
- `runs/` is gitignored — no media files, transcripts, or generated
  output are tracked in version control.
- Every stage caches to disk under `runs/<name>/`; re-running the same
  command after an interruption resumes from the last completed stage
  instead of starting over.
