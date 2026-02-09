"""Text-to-speech using ElevenLabs API."""

import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs


def get_client() -> ElevenLabs:
    """Get an ElevenLabs client using the API key from environment or .env file."""
    load_dotenv()

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError(
            "ELEVENLABS_API_KEY not set. Either:\n"
            "  1. Set environment variable: export ELEVENLABS_API_KEY=...\n"
            "  2. Add to .env file: ELEVENLABS_API_KEY=..."
        )
    return ElevenLabs(api_key=api_key)


# Default voice IDs (standard ElevenLabs voices)
VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",  # Calm, professional female
    "domi": "AZnzlk1XvdvUeBnXmlld",     # Young female
    "bella": "EXAVITQu4vr4xnSDxMaL",    # Soft female
    "antoni": "ErXwobaYiN019PkySvjV",   # Warm male
    "josh": "TxGEqnHWrfWFTfGW9XjX",     # Deep male
    "adam": "pNInz6obpgDQGcFmaJgB",     # Deep male narrator
    "sam": "yoZ06aMxZJJ28mfd3POQ",      # Young male
}


def generate_audio(
    text: str,
    output_path: Path,
    voice: str = "rachel",
    model: str = "eleven_multilingual_v2",
) -> Path:
    """Generate audio from text using ElevenLabs."""
    client = get_client()
    voice_id = VOICES.get(voice.lower(), voice)

    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model,
    )

    audio_bytes = b"".join(audio_generator)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)

    return output_path


def generate_podcast_script(paper_markdown: str, title: str) -> str:
    """Generate an extended two-host podcast discussion for PhD qual prep.

    Creates a rigorous, section-by-section walkthrough followed by
    critical analysis tailored for qualifying exam preparation.

    Args:
        paper_markdown: The cleaned paper markdown.
        title: Paper title.

    Returns:
        A dialogue script with HOST_A and HOST_B labels.
    """
    import anthropic
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Create an extended podcast discussion between two researchers reviewing this academic paper. This is for PhD qualifying exam preparation.

FORMAT:
- Two hosts: HOST_A and HOST_B
- Each line starts with "HOST_A:" or "HOST_B:"
- Natural, conversational academic discussion - hosts should react genuinely ("hmm, interesting", "wait, let me make sure I understand this", "that's a good point"), ask each other clarifying questions, and occasionally push back
- Include natural verbal transitions and breathing room - this should sound like real people talking, not reading a script

This is an extended, in-depth technical episode (target: 20-30 minutes when read aloud). Do NOT rush. Take your time with each section.

LISTENER CONTEXT:
The listener is a PhD candidate in geosciences at UT Austin with a strong engineering and remote sensing background (aerospace engineering, statistical estimation, ICESat-2, satellite stereophotogrammetry, SDB methods). They are preparing for their doctoral qualifying exam with a committee that includes expertise in glaciology/coastal geomorphology, coastal engineering (USACE), fluvial/sedimentary geomorphology, and marine geophysics. Assume the listener is comfortable with remote sensing methodology, uncertainty quantification, and signal processing, but may need more scaffolding on geomorphological processes, sediment transport physics, and earth science conventions.

STRUCTURE — follow this order strictly:

PART 1: FAITHFUL WALKTHROUGH (~15 min)
Walk through the paper section by section, staying close to what the authors actually wrote. For each major section:
- Summarize the key content and findings as presented
- When equations or methods appear, explain them precisely — what each term means physically, what assumptions are embedded, what the inputs and outputs are
- When figures or results are referenced, describe what they show and why the authors included them
- Flag any terminology or geomorphological concepts that an engineering-trained researcher might not have encountered (e.g., depth of closure, Bruun rule, littoral cells, Exner equation context, thermoabrasion vs. thermodenudation) and briefly define them in context
- Do NOT editorialize or critique during this section — just make sure the listener deeply understands what the paper says

PART 2: CRITICAL ANALYSIS & COMMITTEE PREP (~10-15 min)
Now shift to critical evaluation. Address:

a) METHODOLOGICAL SCRUTINY: What are the key assumptions that could be challenged? Where are the error budgets weakest? How sensitive are the conclusions to specific processing choices? A committee member from USACE coastal engineering will probe validation rigor and operational applicability. A geomorphologist will ask whether the methods capture the right physical processes.

b) LIMITATIONS & GAPS: What does the paper NOT address that it probably should? What are the boundary conditions where this approach breaks down? What temporal or spatial scales are missing?

c) CONNECTIONS TO THE LISTENER'S WORK: The listener's dissertation spans Arctic coastal bluff retreat from ArcticDEM stereophotogrammetry, multi-source satellite bathymetric fusion using spacetime kriging at Duck NC, and proposed topobathymetric change detection on the Texas Gulf Coast. Discuss how this paper's methods, findings, or limitations connect to or inform those three chapters.

d) LIKELY COMMITTEE QUESTIONS: Generate 4-5 specific, tough questions that a qualifying exam committee with the described expertise would ask about this paper. For each, briefly outline how the listener should approach answering it.

STYLE NOTES:
- When explaining geomorphology or sediment dynamics concepts, use physical intuition and analogies rather than just definitions
- Be precise about numbers — retreat rates, uncertainties, depth ranges, spatial scales. The committee will expect quantitative fluency.
- If the paper makes claims that conflict with or complement the listener's own published work on Arctic coastal retreat or satellite bathymetry, call that out explicitly
- Do not pad with filler, generic praise, or surface-level observations. Every sentence should either build understanding or sharpen exam readiness.
- Use verbal cues to help the listener follow along: "So moving to the methods section...", "Now let's look at Figure 3...", "Okay, shifting gears to the critical analysis..."

Paper title: {title}

Paper content:
{paper_markdown}

Generate the full discussion:"""

    print("  Calling Claude API for script generation...")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def parse_dialogue(script: str) -> list[tuple[str, str]]:
    """Parse a dialogue script into speaker/text pairs.

    Args:
        script: Script with HOST_A: and HOST_B: labels.

    Returns:
        List of (speaker, text) tuples.
    """
    lines = []
    current_speaker = None
    current_text = []

    for line in script.split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(HOST_[AB]):\s*(.*)$', line)
        if match:
            if current_speaker and current_text:
                lines.append((current_speaker, ' '.join(current_text)))
            current_speaker = match.group(1)
            current_text = [match.group(2)] if match.group(2) else []
        elif current_speaker:
            current_text.append(line)

    if current_speaker and current_text:
        lines.append((current_speaker, ' '.join(current_text)))

    return lines


def set_mp3_metadata(
    mp3_path: Path,
    title: str,
    artist: str = "Paper Podcast",
    album: str = "PhD Qual Prep",
    year: str = None,
    comment: str = None,
) -> None:
    """Set ID3 metadata on an MP3 file for display in music players.

    Args:
        mp3_path: Path to the MP3 file.
        title: Track title (paper title).
        artist: Artist name.
        album: Album name.
        year: Year string.
        comment: Optional comment.
    """
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


def generate_dialogue_audio(
    script: str,
    output_path: Path,
    title: str,
    voice_a: str = "josh",
    voice_b: str = "rachel",
    model: str = "eleven_multilingual_v2",
) -> Path:
    """Generate audio from a two-host dialogue script.

    Args:
        script: Dialogue script with HOST_A/HOST_B labels.
        output_path: Path to save the combined audio file.
        title: Paper title for MP3 metadata.
        voice_a: Voice for Host A.
        voice_b: Voice for Host B.
        model: ElevenLabs model to use.

    Returns:
        Path to the saved audio file.
    """
    client = get_client()

    # Parse the dialogue
    print("  Parsing dialogue script...")
    dialogue = parse_dialogue(script)
    if not dialogue:
        raise ValueError("Could not parse dialogue from script")

    print(f"  Found {len(dialogue)} dialogue segments")

    # Generate audio for each segment
    print("  Generating audio segments...")
    audio_segments = []
    total = len(dialogue)

    for i, (speaker, text) in enumerate(dialogue):
        voice = voice_a if speaker == "HOST_A" else voice_b
        voice_id = VOICES.get(voice.lower(), voice)

        progress = f"[{i+1}/{total}]"
        speaker_name = "Host A" if speaker == "HOST_A" else "Host B"
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"    {progress} {speaker_name}: {preview}")

        audio_generator = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model,
        )
        audio_bytes = b"".join(audio_generator)
        audio_segments.append(audio_bytes)

    # Combine all segments
    print("  Combining audio segments...")
    combined_audio = b"".join(audio_segments)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(combined_audio)

    # Set MP3 metadata
    print("  Setting MP3 metadata...")
    set_mp3_metadata(
        output_path,
        title=title,
        artist="Paper Podcast",
        album="PhD Qual Prep",
        comment=f"Generated {datetime.now().strftime('%Y-%m-%d')}"
    )

    return output_path


def create_podcast(
    paper_markdown: str,
    title: str,
    output_dir: Path,
    voice_a: str = "josh",
    voice_b: str = "rachel",
) -> dict:
    """Full pipeline: generate script and audio from paper markdown.

    Args:
        paper_markdown: Cleaned paper markdown.
        title: Paper title.
        output_dir: Directory to save outputs.
        voice_a: Voice for Host A.
        voice_b: Voice for Host B.

    Returns:
        Dictionary with paths to script and audio files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate script
    print("\n[1/3] Generating podcast script...")
    script = generate_podcast_script(paper_markdown, title)
    script_path = output_dir / "podcast_script.txt"
    script_path.write_text(script)
    print(f"  Script: {len(script):,} characters")
    print(f"  Saved to: {script_path}")

    # Generate audio
    print("\n[2/3] Generating podcast audio...")

    # Create safe filename from title
    safe_title = re.sub(r'[^\w\s-]', '', title)
    safe_title = re.sub(r'\s+', '_', safe_title)[:60]
    audio_filename = f"{safe_title}.mp3"
    audio_path = output_dir / audio_filename

    audio_path = generate_dialogue_audio(
        script,
        audio_path,
        title=title,
        voice_a=voice_a,
        voice_b=voice_b,
    )

    size_mb = audio_path.stat().st_size / 1024 / 1024
    print(f"\n[3/3] Complete!")
    print(f"  Audio: {audio_path.name} ({size_mb:.1f} MB)")

    return {
        "script_path": script_path,
        "audio_path": audio_path,
        "script": script,
    }


# Test when run directly
if __name__ == "__main__":
    print("Available voices:")
    for name, vid in VOICES.items():
        print(f"  - {name}: {vid}")
