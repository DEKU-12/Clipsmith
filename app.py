"""Streamlit UI for Clipsmith — paste a URL, run the pipeline, browse results.

Wraps run_pipeline.py as a subprocess so the CLI stays the source of
truth; this is purely a viewer/trigger layer on top of it.
"""

import json
import os
import subprocess
import sys
import time

import streamlit as st

st.set_page_config(page_title="Clipsmith", page_icon="🎬", layout="wide")

st.title("🎬 Clipsmith")
st.caption("Paste a YouTube URL, get captioned short-form clips + a newsletter draft.")

with st.form("run_form"):
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    col1, col2, col3 = st.columns(3)
    with col1:
        clips = st.number_input("Number of clips", min_value=1, max_value=15, value=5)
    with col2:
        max_minutes = st.number_input("Max minutes (0 = full episode)", min_value=0.0, value=0.0, step=1.0)
    with col3:
        dry_run = st.checkbox("Dry run (no API calls, test only)", value=False)
    submitted = st.form_submit_button("Run pipeline")

if submitted:
    if not url.strip():
        st.error("Please paste a YouTube URL.")
        st.stop()

    workdir = f"runs/ui_{int(time.time())}"
    cmd = [sys.executable, "run_pipeline.py", url.strip(), "--clips", str(int(clips)), "--workdir", workdir]
    if max_minutes > 0:
        cmd += ["--max-minutes", str(max_minutes)]
    if dry_run:
        cmd.append("--dry-run")

    log_lines = []
    with st.status("Running pipeline...", expanded=True) as status:
        log_box = st.empty()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            log_lines.append(line.rstrip())
            log_box.code("\n".join(log_lines[-40:]))
        process.wait()

        if process.returncode != 0:
            status.update(label="Pipeline failed", state="error")
            st.error("Pipeline failed. See log above for details.")
            st.stop()
        status.update(label="Done", state="complete")

    st.session_state["last_workdir"] = workdir

if "last_workdir" in st.session_state:
    outputs_dir = os.path.join(st.session_state["last_workdir"], "outputs")
    metadata_path = os.path.join(outputs_dir, "metadata.json")
    newsletter_path = os.path.join(outputs_dir, "newsletter.md")

    if os.path.exists(metadata_path):
        st.header("Clips")
        with open(metadata_path) as f:
            metadata = json.load(f)

        for i, clip in enumerate(metadata, start=1):
            with st.container(border=True):
                col_video, col_info = st.columns([1, 2])
                clip_path = os.path.join(outputs_dir, clip["file"])
                with col_video:
                    if os.path.exists(clip_path):
                        st.video(clip_path)
                with col_info:
                    st.subheader(f'{i}. {clip["title"]} · score {clip["score"]}')
                    st.write(f'**Hook:** {clip["hook_line"]}')
                    st.write(clip["description"])
                    st.write(" ".join(f"#{h.lstrip('#')}" for h in clip["hashtags"]))
                    st.caption(f'{clip["start"]:.1f}s – {clip["end"]:.1f}s · {clip["reason"]}')

    if os.path.exists(newsletter_path):
        st.header("Newsletter draft")
        with open(newsletter_path) as f:
            st.markdown(f.read())
