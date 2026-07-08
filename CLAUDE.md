# Clipsmith — project context

AI pipeline: long-form YouTube episode in → captioned 9:16 shorts +
publish metadata + newsletter draft out. Built as a working demo — it
must actually run end to end, and console output quality matters (the
per-clip scoring printout is the demo centerpiece).

## Layout

- `run_pipeline.py` — CLI + orchestrator; every stage lives here
- `prompts.py` — all LLM prompt templates (find-moments, score-clip, newsletter)
- `captions.py` — word timestamps → styled `.ass` karaoke subtitle file
- `facecrop.py` — face detection → 9:16 crop x-offset (YuNet model vendored as `face_detection_yunet_2023mar.onnx`)
- `app.py` — optional Streamlit UI; shells out to `run_pipeline.py`, never bypasses it
- `runs/<name>/` — per-episode working dir, gitignored, holds all caches and outputs

## Conventions

- Plain Python scripts, no framework, no packages/classes unless something
  is actually broken — demo velocity over architecture purity.
- Every pipeline stage caches its output to disk and skips if the file
  already exists. Re-running resumes from the last completed stage.
  When changing a stage, delete its cached artifact in `runs/<name>/`
  to force regeneration.
- LLM: Groq `llama-3.3-70b-versatile` via `GROQ_API_KEY` env var (free
  tier; swapped in for the originally-planned Claude API to keep demo
  runs free). All LLM calls expect pure-JSON responses; strip ``` fences
  before parsing, retry once on a parse failure, validate/default fields
  after.
- NEVER write any API key into a file. Keys live in the shell env only.

## Testing

- Cheap end-to-end test (no LLM calls, no API key needed):
  `python run_pipeline.py <url> --dry-run --max-minutes 3 --clips 2`
- Real run: `python run_pipeline.py <url> --max-minutes 18 --clips 5`
- CI (GitHub Actions) runs the dry-run against a synthetic ffmpeg-generated
  video — YouTube blocks downloads from CI runner IPs, so never make the
  workflow fetch a real video.
- Verify caption/crop changes visually: extract a frame with
  `ffmpeg -ss N -i short_01.mp4 -frames:v 1 frame.jpg` and look at it.

## Gotchas

- ffmpeg must be built with libass (`brew install homebrew-ffmpeg/ffmpeg/ffmpeg`);
  the default Homebrew formula lacks the `ass` filter and caption burning
  dies with "No option name near ...".
- yt-dlp calls need `--no-playlist` and `--` before the URL (radio-mix
  URLs otherwise enumerate a whole playlist; `-`-prefixed input would
  parse as flags).
- Cuts must never start/end mid-word: clip boundaries snap to sentence
  starts/ends (±0.3s pad) and clips are clamped to 20–75s.
- Don't commit anything in `runs/`, any media file, or `.env`.
