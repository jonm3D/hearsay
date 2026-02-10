"""Text-to-speech using Kokoro TTS (local, free)."""

import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment

# Kokoro sample rate
SAMPLE_RATE = 24000

# Default narrator voice
DEFAULT_VOICE = "af_heart"

# Lazy-loaded pipeline singleton
_pipeline = None


def _get_pipeline():
    """Get or create the Kokoro TTS pipeline (lazy singleton).

    First call downloads the model (~300MB) from HuggingFace and takes a few seconds.
    Subsequent calls return the cached pipeline instantly.
    """
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        print("  Loading Kokoro TTS model...")
        _pipeline = KPipeline(lang_code='a')  # American English
    return _pipeline


def _get_client():
    """Get an Anthropic client."""
    import anthropic
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def _build_script_prompt(paper_markdown: str, title: str) -> str:
    """Build the prompt for generating a narration script."""
    return f"""Create a single-narrator audio summary of this academic paper. This is for PhD qualifying exam preparation — meant to be listened to, not read.

FORMAT:
- Single narrator speaking directly to the listener ("you")
- No dialogue, no fake hosts, no performative reactions
- Written for speech: short sentences, verbal signposts, no parentheticals or nested clauses
- Spell out abbreviations on first use, avoid "as shown in Table 3" (the listener can't see it — describe what the table shows)
- Use natural pacing cues: "Alright, moving on to methods." / "This next part is important." / "Let's slow down here."

TARGET LENGTH: ~10 minutes when read aloud at natural pace.

LISTENER CONTEXT:
The listener is a PhD candidate in geosciences at UT Austin with a strong engineering and remote sensing background (aerospace engineering, statistical estimation, ICESat-2, satellite stereophotogrammetry, SDB methods). They are preparing for their doctoral qualifying exam with a committee that includes expertise in glaciology/coastal geomorphology, coastal engineering (USACE), fluvial/sedimentary geomorphology, and marine geophysics. Assume the listener is comfortable with remote sensing methodology, uncertainty quantification, and signal processing, but may need more scaffolding on geomorphological processes, sediment transport physics, and earth science conventions.

FIDELITY RULE:
Stay faithful to what the paper actually says. Do not invent connections the authors didn't make, do not fabricate citations, and do not attribute claims to the paper that aren't there. If you contextualize beyond the paper's content, signal it clearly: "The paper doesn't address this directly, but for your committee prep, it's worth knowing that..."

STRUCTURE — follow this order strictly:

PART 1: PAPER WALKTHROUGH (~6-7 min)
Walk through the paper section by section, closely tracking the authors' own structure. For each major section:
- Summarize the key content and findings as presented
- When equations or methods appear, explain them precisely — what each term means physically, what assumptions are embedded, what the inputs and outputs are
- When figures or results are discussed, describe what they show and why they matter, in enough detail that the listener can reconstruct the key takeaway without seeing the figure
- Flag terminology or geomorphological concepts that an engineering-trained researcher might not have encountered (e.g., depth of closure, Bruun rule, littoral cells, Exner equation context, thermoabrasion vs. thermodenudation) and define them with physical intuition, not just textbook definitions
- Do NOT critique during this section — the goal is accurate comprehension

PART 2: CRITICAL ANALYSIS & COMMITTEE PREP (~3-4 min)
Shift explicitly: "Alright, now that we've covered what the paper says, let's talk about what a committee would push on."

a) METHODOLOGICAL SCRUTINY: Key assumptions that could be challenged. Where the error budgets are weakest. Sensitivity of conclusions to specific processing choices. Think about what a USACE coastal engineer would probe on validation and operational applicability, and what a geomorphologist would ask about whether the methods capture the right physical processes.

b) LIMITATIONS & GAPS: What the paper doesn't address that it probably should. Boundary conditions where the approach breaks down. Missing temporal or spatial scales.

c) CONNECTIONS TO THE LISTENER'S WORK: The listener's dissertation spans Arctic coastal bluff retreat from ArcticDEM stereophotogrammetry, multi-source satellite bathymetric fusion using spacetime kriging at Duck NC, and proposed topobathymetric change detection on the Texas Gulf Coast. Where this paper's methods, findings, or limitations connect to or inform those three chapters. Only make connections that are substantive — skip anything that would be a stretch.

d) COMMITTEE QUESTIONS: 4-5 specific, tough questions the described committee would likely ask about this paper. For each, give a concise sketch of how to approach the answer.

STYLE NOTES:
- Be precise about numbers — retreat rates, uncertainties, depth ranges, spatial scales
- Prioritize density of insight over length. No filler, no generic praise, no "this is a really important paper"
- If a section of the paper is routine or standard, summarize it briefly and move on. Spend time where the intellectual content is
- Write for the ear: prefer "about 3 meters per year" over "approximately 3 m/yr", prefer "plus or minus half a meter" over "±0.5 m"

Paper title: {title}

Paper content:
{paper_markdown}

Generate the full narrated summary:"""


def generate_script(paper_markdown: str, title: str) -> str:
    """Generate a single-narrator audio summary for PhD qual prep.

    Args:
        paper_markdown: The cleaned paper markdown.
        title: Paper title.

    Returns:
        Plain narration text (no dialogue labels).
    """
    client = _get_client()
    prompt = _build_script_prompt(paper_markdown, title)

    print("  Calling Claude API for script generation...")
    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def set_mp3_metadata(
    mp3_path: Path,
    title: str,
    artist: str = "Hearsay",
    album: str = "PhD Qual Prep",
    year: str = None,
    comment: str = None,
) -> None:
    """Set ID3 metadata on an MP3 file for display in music players."""
    from mutagen.id3 import ID3, TIT2, TPE1, TALB, TYER, COMM, ID3NoHeaderError

    mp3_path = Path(mp3_path)

    try:
        tags = ID3(mp3_path)
    except ID3NoHeaderError:
        tags = ID3()

    tags.add(TIT2(encoding=3, text=title))
    tags.add(TPE1(encoding=3, text=artist))
    tags.add(TALB(encoding=3, text=album))

    if year:
        tags.add(TYER(encoding=3, text=year))
    else:
        tags.add(TYER(encoding=3, text=str(datetime.now().year)))

    if comment:
        tags.add(COMM(encoding=3, lang='eng', desc='', text=comment))

    tags.save(mp3_path)


def _synthesize_segment(pipeline, text: str, voice: str) -> np.ndarray:
    """Synthesize a single text segment and return concatenated audio.

    Kokoro's pipeline yields chunks; we concatenate them into one array.
    """
    chunks = []
    for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=1.2):
        chunks.append(audio)

    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(chunks)


def generate_audio(
    script: str,
    output_path: Path,
    title: str,
    voice: str = DEFAULT_VOICE,
) -> Path:
    """Generate audio from a narration script using Kokoro TTS.

    Splits the script into paragraphs, synthesizes each, and combines
    them into a single MP3 with brief pauses between paragraphs.

    Args:
        script: Plain narration text.
        output_path: Path to save the MP3 file.
        title: Paper title for MP3 metadata.
        voice: Kokoro voice name.

    Returns:
        Path to the saved audio file.
    """
    pipeline = _get_pipeline()

    # Split into paragraphs for progress reporting and natural pauses
    paragraphs = [p.strip() for p in script.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [script.strip()]

    print(f"  Synthesizing {len(paragraphs)} paragraphs...")

    # Brief pause between paragraphs (0.4s)
    pause = np.zeros(int(SAMPLE_RATE * 0.4), dtype=np.float32)

    audio_segments = []
    for i, para in enumerate(paragraphs):
        preview = para[:60] + "..." if len(para) > 60 else para
        print(f"    [{i+1}/{len(paragraphs)}] {preview}")

        segment = _synthesize_segment(pipeline, para, voice)
        if segment.size > 0:
            audio_segments.append(segment)
            audio_segments.append(pause)

    if not audio_segments:
        raise ValueError("No audio was generated")

    # Combine
    print("  Combining audio...")
    combined = np.concatenate(audio_segments)

    duration_min = len(combined) / SAMPLE_RATE / 60
    print(f"  Total duration: {duration_min:.1f} minutes")

    # WAV -> MP3
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    try:
        sf.write(str(tmp_wav), combined, SAMPLE_RATE)
        audio_seg = AudioSegment.from_wav(str(tmp_wav))
        audio_seg.export(str(output_path), format="mp3", bitrate="192k")
    finally:
        tmp_wav.unlink(missing_ok=True)

    # Metadata
    print("  Setting MP3 metadata...")
    set_mp3_metadata(
        output_path,
        title=title,
        artist="Hearsay",
        album="PhD Qual Prep",
        comment=f"Generated {datetime.now().strftime('%Y-%m-%d')} with Kokoro TTS",
    )

    return output_path


def create_podcast(
    paper_markdown: str,
    title: str,
    output_dir: Path,
    voice: str = DEFAULT_VOICE,
) -> dict:
    """Full pipeline: stream script from Claude while synthesizing audio.

    Streams the Claude response paragraph-by-paragraph and feeds completed
    paragraphs to a TTS worker thread, so generation and synthesis overlap.

    Args:
        paper_markdown: Cleaned paper markdown.
        title: Paper title.
        output_dir: Directory to save outputs.
        voice: Kokoro voice name for the narrator.

    Returns:
        Dictionary with paths to script and audio files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load TTS model so it's ready when first paragraph arrives
    pipeline = _get_pipeline()

    client = _get_client()
    prompt = _build_script_prompt(paper_markdown, title)

    print("\nStreaming script + synthesizing audio...")

    # Single TTS worker: synthesizes paragraphs in order while Claude streams
    executor = ThreadPoolExecutor(max_workers=1)
    tts_futures = []
    script_parts = []
    buffer = ""
    para_idx = 0

    def _submit_paragraph(text: str):
        nonlocal para_idx
        para_idx += 1
        preview = text[:60] + "..." if len(text) > 60 else text
        print(f"  [paragraph {para_idx}] {preview}")
        script_parts.append(text)
        future = executor.submit(_synthesize_segment, pipeline, text, voice)
        tts_futures.append(future)

    with client.messages.stream(
        model="claude-opus-4-20250514",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            buffer += chunk
            # Emit complete paragraphs as they arrive
            while "\n\n" in buffer:
                paragraph, buffer = buffer.split("\n\n", 1)
                paragraph = paragraph.strip()
                if paragraph:
                    _submit_paragraph(paragraph)

    # Flush remaining buffer
    if buffer.strip():
        _submit_paragraph(buffer.strip())

    print(f"  Script complete: {para_idx} paragraphs")

    # Save script
    script = "\n\n".join(script_parts)
    script_path = output_dir / "script.txt"
    script_path.write_text(script)

    # Collect audio segments in order (blocks until each future completes)
    pause = np.zeros(int(SAMPLE_RATE * 0.4), dtype=np.float32)
    audio_segments = []
    for i, future in enumerate(tts_futures):
        segment = future.result()
        if segment.size > 0:
            audio_segments.append(segment)
            audio_segments.append(pause)
        print(f"  [audio {i + 1}/{len(tts_futures)}] done")

    executor.shutdown(wait=False)

    if not audio_segments:
        raise ValueError("No audio was generated")

    # Combine and export
    print("  Combining audio...")
    combined = np.concatenate(audio_segments)
    duration_min = len(combined) / SAMPLE_RATE / 60
    print(f"  Total duration: {duration_min:.1f} minutes")

    safe_title = re.sub(r'[^\w\s-]', '', title)
    safe_title = re.sub(r'\s+', '_', safe_title)[:60]
    audio_path = output_dir / f"{safe_title}.mp3"
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    try:
        sf.write(str(tmp_wav), combined, SAMPLE_RATE)
        audio_seg = AudioSegment.from_wav(str(tmp_wav))
        audio_seg.export(str(audio_path), format="mp3", bitrate="192k")
    finally:
        tmp_wav.unlink(missing_ok=True)

    # Metadata
    set_mp3_metadata(
        audio_path,
        title=title,
        artist="Hearsay",
        album="PhD Qual Prep",
        comment=f"Generated {datetime.now().strftime('%Y-%m-%d')} with Kokoro TTS",
    )

    size_mb = audio_path.stat().st_size / 1024 / 1024
    print(f"\n  Complete! {audio_path.name} ({size_mb:.1f} MB)")

    return {
        "script_path": script_path,
        "audio_path": audio_path,
        "script": script,
    }


# Test when run directly
if __name__ == "__main__":
    print(f"Default voice: {DEFAULT_VOICE}")
    print("\nQuick synthesis test...")
    test_text = "This is a test of the hearsay narration pipeline. The audio should be clear and natural."
    output = Path("test_narration.mp3")
    generate_audio(test_text, output, "Test Narration")
    print(f"Saved: {output} ({output.stat().st_size / 1024:.0f} KB)")
