# Paper Podcast

Convert academic papers from your Zotero library into podcast-style audio reviews for PhD qualifying exam preparation.

## Overview

Paper Podcast extracts text and figures from PDFs in your Zotero library, uses Claude to clean the content and generate a two-host discussion script, then converts it to audio using ElevenLabs TTS. The result is a podcast episode designed to help you deeply understand papers before your qualifying exam.

## Features

- **Zotero Integration**: Directly reads your Zotero SQLite database to browse collections and select papers
- **Intelligent PDF Extraction**: Extracts text and figures using PyMuPDF with extensive cleaning
- **Claude-Powered Processing**:
  - Cleans and structures extracted text into markdown
  - Describes figures using vision capabilities
  - Filters out ads and artifacts from extracted images
  - Generates in-depth two-host discussion scripts
- **Two-Host Dialogue**: Creates natural conversation between HOST_A and HOST_B with distinct voices
- **PhD Qual Focus**: Prompts are tailored for qualifying exam preparation with:
  - Faithful paper walkthrough (methodology, equations, figures)
  - Critical analysis and limitations
  - Connections to your research
  - Likely committee questions with suggested approaches
- **MP3 Output**: Generates audio files with proper ID3 metadata for music players

## Installation

```bash
# Clone the repository
git clone git@github.com:jonm3D/paper_to_podcast.git
cd paper_to_podcast

# Install in development mode
pip install -e .
```

## Configuration

Create a `.env` file in the project root with your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
ELEVENLABS_API_KEY=...
```

Optionally set your Zotero data directory (defaults to `~/Zotero`):

```
ZOTERO_DATA_DIR=/path/to/Zotero
```

## Usage

### Command Line

```bash
# Process a paper from a specific collection
paper-podcast --collection "Qualifying Exam"

# Search across your entire library
paper-podcast --search "coastal erosion"

# Specify output directory
paper-podcast --collection "Papers" --output-dir ./podcasts
```

### Interactive Selection

When you run the command, you'll be presented with papers in the collection and can select which one to process.

### Python API

```python
from paper_podcast.zotero import get_collections, get_papers_in_collection, search_papers
from paper_podcast.pdf import extract_text_raw, extract_figures
from paper_podcast.review import process_paper
from paper_podcast.tts import create_podcast

# List collections
collections = get_collections()

# Get papers from a collection
papers = get_papers_in_collection("My Collection")

# Search library
results = search_papers("machine learning")

# Process a PDF
markdown = process_paper(pdf_path, title)

# Generate podcast
result = create_podcast(markdown, title, output_dir)
```

## Output Structure

```
output/
└── paper_title/
    ├── podcast_script.txt    # Generated dialogue script
    └── Paper_Title.mp3       # Audio file with ID3 metadata
```

The MP3 includes metadata:
- **Title**: Paper title
- **Artist**: "Paper Podcast"
- **Album**: "PhD Qual Prep"
- **Year**: Generation date
- **Comment**: Generation timestamp

## Project Structure

```
paper-podcasts/
├── pyproject.toml
├── src/
│   └── paper_podcast/
│       ├── __init__.py
│       ├── cli.py        # CLI entry point
│       ├── zotero.py     # Zotero SQLite integration
│       ├── pdf.py        # PDF text/figure extraction
│       ├── review.py     # Claude API for cleaning/figures
│       └── tts.py        # Script generation & ElevenLabs TTS
└── output/               # Default output directory
```

## How It Works

1. **Zotero Query**: Connects to your local Zotero SQLite database (in immutable mode to avoid conflicts) and retrieves papers with their PDF paths

2. **PDF Extraction**: Uses PyMuPDF to extract raw text and images, with preprocessing to handle:
   - Multi-column layouts
   - Broken lines and hyphenation
   - Headers, footers, and page numbers
   - Cover pages and institutional headers

3. **Claude Cleaning**: Sends extracted content to Claude API to:
   - Structure text as clean markdown
   - Describe figures using vision capabilities
   - Filter out advertisements and artifacts

4. **Script Generation**: Claude generates a ~20-30 minute two-host discussion with:
   - Part 1: Faithful walkthrough of the paper
   - Part 2: Critical analysis and exam preparation

5. **Audio Synthesis**: ElevenLabs converts the script to audio:
   - HOST_A: "josh" voice (deep male)
   - HOST_B: "rachel" voice (calm female)
   - Segments are concatenated into a single MP3

## Dependencies

- `click` - CLI framework
- `python-dotenv` - Environment variable loading
- `pymupdf` - PDF text and image extraction
- `anthropic` - Claude API client
- `elevenlabs` - Text-to-speech API
- `mutagen` - MP3 ID3 metadata

## Requirements

- Python 3.10+
- Zotero with local storage (not cloud-only)
- Anthropic API key
- ElevenLabs API key

## Notes

- The tool reads your Zotero database in read-only mode and never modifies it
- Processing a single paper uses approximately 10-20k tokens (Claude) and 15-25k characters (ElevenLabs)
- Audio generation can take several minutes for long discussions
- The podcast prompt is tailored for a geosciences PhD candidate but can be customized in `tts.py`

## License

MIT
