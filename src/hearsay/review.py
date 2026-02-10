"""Anthropic API integration for paper processing."""

import base64
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv


def get_client() -> anthropic.Anthropic:
    """Get an Anthropic client using the API key from environment or .env file."""
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Either:\n"
            "  1. Set environment variable: export ANTHROPIC_API_KEY=sk-...\n"
            "  2. Create .env file with: ANTHROPIC_API_KEY=sk-..."
        )
    return anthropic.Anthropic(api_key=api_key)


def slugify(title: str, max_length: int = 80) -> str:
    """Convert a title to a safe filename/folder slug."""
    slug = re.sub(r"[^\w\s-]", "", title)
    slug = re.sub(r"[\s]+", "_", slug)
    slug = slug[:max_length].strip("_")
    return slug.lower()


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Encode an image to base64 and determine its media type."""
    image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

    suffix = image_path.suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/png")

    return image_data, media_type


def is_paper_figure(image_path: Path, paper_context: str = "") -> bool:
    """Check if an image is an actual paper figure vs an ad/artifact.

    Args:
        image_path: Path to the image.
        paper_context: Context about the paper topic.

    Returns:
        True if this appears to be a real figure, False if it's an ad/artifact.
    """
    client = get_client()
    image_data, media_type = _encode_image(image_path)

    prompt = f"""Is this image a scientific figure from an academic paper, or is it an advertisement, journal logo, conference announcement, or other non-figure artifact?

Reply with ONLY one word: "figure" or "artifact"

{f"Paper topic: {paper_context}" if paper_context else ""}"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    response = message.content[0].text.strip().lower()
    return "figure" in response


def describe_figure(image_path: Path, figure_num: int, paper_context: str = "") -> str:
    """Use Claude's vision to describe a figure from the paper.

    Args:
        image_path: Path to the figure image.
        figure_num: The figure number for reference.
        paper_context: Optional context about the paper topic.

    Returns:
        A description of the figure suitable for a podcast.
    """
    client = get_client()
    image_data, media_type = _encode_image(image_path)

    prompt = f"""Describe this figure (Figure {figure_num}) from an academic paper in 2-4 sentences.
Focus on:
- What type of visualization it is (map, graph, scatter plot, diagram, etc.)
- The key information it conveys
- Any notable patterns or findings visible

Keep the description clear and suitable for a podcast listener who cannot see the image.
{f"Paper context: {paper_context}" if paper_context else ""}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    return message.content[0].text


def _chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _clean_chunk(client, chunk: str, chunk_num: int, total_chunks: int,
                 title: str | None = None) -> str:
    """Clean a single chunk of raw PDF text."""
    position = f"Part {chunk_num}/{total_chunks}. "
    prompt = f"""Clean this section of raw PDF text into readable markdown. {position}Return ONLY the cleaned content.

Instructions:
- Remove repository metadata, download notices, copyright footers, page numbers
- Remove journal formatting artifacts (column breaks, page headers)
- Fix broken words/sentences from PDF column layouts
- Use proper markdown: ## for section headings, paragraphs separated by blank lines
- Keep all academic content, citations, equations, figure/table references
- Do NOT summarize or modify the content - just clean and format it

{"Title: " + title if title else ""}

Raw PDF text:
{chunk}"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def clean_paper_text(raw_text: str, title: str | None = None) -> str:
    """Clean raw PDF text into readable markdown using parallel Haiku calls.

    Chunks the text and cleans each chunk concurrently for speed.

    Args:
        raw_text: Raw text extracted from PDF (may have artifacts).
        title: Optional paper title for context.

    Returns:
        Clean markdown text of just the paper content.
    """
    from concurrent.futures import ThreadPoolExecutor

    client = get_client()
    chunks = _chunk_text(raw_text)
    print(f"    Cleaning {len(chunks)} chunks in parallel...")

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(_clean_chunk, client, chunk, i + 1, len(chunks), title)
            for i, chunk in enumerate(chunks)
        ]
        results = [f.result() for f in futures]

    return "\n\n".join(results)


def process_paper(
    pdf_path: Path,
    title: str,
    output_dir: Path,
    extract_figures: bool = True,
    describe_figures: bool = True,
) -> dict:
    """Process a paper: extract text, figures, and create clean markdown.

    Creates folder structure:
        output_dir/<paper-slug>/
        ├── paper.md
        └── img/
            ├── figure_1.png
            ├── figure_2.jpg
            └── ...

    Args:
        pdf_path: Path to the PDF file.
        title: Paper title.
        output_dir: Base output directory.
        extract_figures: Whether to extract figures from PDF.
        describe_figures: Whether to generate AI descriptions of figures.

    Returns:
        Dictionary with paths and metadata.
    """
    from hearsay.pdf import extract_text_raw, extract_figures as pdf_extract_figures

    # Create paper folder
    paper_slug = slugify(title)
    paper_dir = Path(output_dir) / paper_slug
    paper_dir.mkdir(parents=True, exist_ok=True)

    img_dir = paper_dir / "img"
    img_dir.mkdir(exist_ok=True)

    result = {
        "paper_dir": paper_dir,
        "title": title,
        "figures": [],
        "figure_descriptions": {},
    }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Extract raw text and figures (both fast, local)
    print(f"  Extracting text...")
    raw_text = extract_text_raw(pdf_path)

    figure_paths = []
    all_images = []
    if extract_figures:
        print(f"  Extracting figures...")
        all_images = pdf_extract_figures(pdf_path, img_dir)
        print(f"  Found {len(all_images)} images")

    # Run text cleaning and figure processing in parallel
    # Text cleaning is the bottleneck — start it ASAP alongside figure API calls
    print(f"  Running text cleaning + figure processing in parallel...")

    with ThreadPoolExecutor(max_workers=8) as pool:
        # Submit text cleaning (internally parallelized into chunks)
        text_future = pool.submit(clean_paper_text, raw_text, title)

        # Submit all figure filter calls in parallel
        if all_images:
            filter_futures = {
                pool.submit(is_paper_figure, img, title): img
                for img in all_images
            }

            for future in as_completed(filter_futures):
                img_path = filter_futures[future]
                try:
                    if future.result():
                        figure_paths.append(img_path)
                    else:
                        print(f"    Filtered out artifact: {img_path.name}")
                        img_path.unlink()
                except Exception as e:
                    print(f"    Warning: Could not classify {img_path.name}: {e}")
                    figure_paths.append(img_path)

            # Sort by original order and rename sequentially
            figure_paths.sort(key=lambda p: all_images.index(p))
            final_paths = []
            for i, fig_path in enumerate(figure_paths, 1):
                new_name = f"figure_{i}{fig_path.suffix}"
                new_path = fig_path.parent / new_name
                if fig_path != new_path:
                    fig_path.rename(new_path)
                final_paths.append(new_path)
            figure_paths = final_paths
            result["figures"] = figure_paths
            print(f"  Kept {len(figure_paths)} figures")

        # Submit figure descriptions in parallel (after filtering)
        if describe_figures and figure_paths:
            print(f"  Describing {len(figure_paths)} figures in parallel...")
            desc_futures = {
                pool.submit(describe_figure, fig_path, i, title): (i, fig_path)
                for i, fig_path in enumerate(figure_paths, 1)
            }
            for future in as_completed(desc_futures):
                i, fig_path = desc_futures[future]
                try:
                    result["figure_descriptions"][f"figure_{i}"] = future.result()
                    print(f"    Figure {i} described")
                except Exception as e:
                    print(f"    Warning: Could not describe figure {i}: {e}")
                    result["figure_descriptions"][f"figure_{i}"] = f"[Figure {i}]"

        # Wait for text cleaning to finish
        clean_text = text_future.result()
        print(f"  Text cleaning done ({len(clean_text):,} chars)")

    # Insert figure descriptions into the markdown
    if result["figure_descriptions"]:
        clean_text = _insert_figure_descriptions(clean_text, result["figure_descriptions"], figure_paths)

    # Save markdown
    md_path = paper_dir / "paper.md"
    md_path.write_text(clean_text, encoding="utf-8")
    result["markdown_path"] = md_path

    print(f"  Saved to: {paper_dir}/")

    return result


def _insert_figure_descriptions(
    markdown: str, descriptions: dict[str, str], figure_paths: list[Path]
) -> str:
    """Insert figure descriptions and image references into markdown.

    Looks for figure references like "Figure 1" or "Fig. 1" and adds
    the AI-generated description and image path after them.
    """
    # Add a figures section at the end if not present
    figures_section = "\n\n## Figures\n\n"

    for i, (key, desc) in enumerate(descriptions.items(), 1):
        fig_num = i
        if figure_paths and i <= len(figure_paths):
            img_path = figure_paths[i - 1]
            rel_path = f"img/{img_path.name}"
            figures_section += f"### Figure {fig_num}\n\n"
            figures_section += f"![Figure {fig_num}]({rel_path})\n\n"
            figures_section += f"**Description:** {desc}\n\n"
        else:
            figures_section += f"### Figure {fig_num}\n\n"
            figures_section += f"**Description:** {desc}\n\n"

    # Insert before References section if it exists, otherwise at end
    if "## References" in markdown:
        markdown = markdown.replace("## References", figures_section + "## References")
    else:
        markdown += figures_section

    return markdown


# Test when run directly
if __name__ == "__main__":
    from hearsay.zotero import get_papers_in_collection

    papers = get_papers_in_collection("Texas Coast")
    paper = [p for p in papers if p.pdf_path and "ICESat-2" in p.title][0]

    print(f"Processing: {paper.title}")
    print()

    result = process_paper(
        pdf_path=paper.pdf_path,
        title=paper.title,
        output_dir=Path("output"),
        extract_figures=True,
        describe_figures=True,
    )

    print()
    print(f"Done! Output in: {result['paper_dir']}")
    print(f"Figures: {len(result['figures'])}")
    print(f"Markdown: {result['markdown_path']}")
