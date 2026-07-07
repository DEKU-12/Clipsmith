"""Word-timestamps -> styled .ass karaoke subtitle file."""

import os

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FONT_SIZE = 90
CHUNK_SIZE = 3
# Distance from the bottom edge, positions captions in the lower-middle third.
MARGIN_V = 420

ASS_HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {VIDEO_WIDTH}
PlayResY: {VIDEO_HEIGHT}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial Black,{FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,6,0,2,60,60,{MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _chunk(words, size):
    for i in range(0, len(words), size):
        yield words[i : i + size]


def build_ass_file(words: list[dict], output_path: str, chunk_size: int = CHUNK_SIZE) -> str:
    """words: list of {"word": str, "start": float, "end": float}, all
    timestamps relative to the clip's own start (i.e. already shifted so
    the first word is near t=0). Writes an .ass file to output_path.
    """
    lines = [ASS_HEADER]
    prev_chunk_end = 0.0

    for chunk in _chunk(words, chunk_size):
        if not chunk:
            continue

        chunk_start = max(chunk[0]["start"], prev_chunk_end)
        chunk_end = max(chunk[-1]["end"], chunk_start + 0.1)

        karaoke_text = ""
        for w in chunk:
            w_start = max(w["start"], chunk_start)
            w_end = max(w["end"], w_start + 0.01)
            duration_cs = max(1, round((w_end - w_start) * 100))
            karaoke_text += f"{{\\k{duration_cs}}}{w['word'].upper()} "

        lines.append(
            f"Dialogue: 0,{_fmt_time(chunk_start)},{_fmt_time(chunk_end)},"
            f"Karaoke,,0,0,0,,{karaoke_text.strip()}"
        )
        prev_chunk_end = chunk_end

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return output_path
