"""Command-line interface for hearsay."""

from pathlib import Path

import click


@click.command()
@click.option("--collection", "-c", default=None, help="Browse papers in a Zotero collection.")
@click.option("--search", "-s", default=None, help="Search papers by title across the whole library.")
@click.option("--output-dir", "-o", default="./output", type=click.Path(), help="Output directory.")
@click.option("--no-figures", is_flag=True, help="Skip figure extraction.")
@click.option("--no-audio", is_flag=True, help="Generate script only, skip TTS.")
def main(collection, search, output_dir, no_figures, no_audio):
    """Convert academic papers from Zotero into audio reviews."""
    from hearsay.zotero import get_collections, get_papers_in_collection, search_papers

    output_dir = Path(output_dir)

    # Step 1: Get papers — from collection, search, or interactive pick
    if collection:
        try:
            papers = get_papers_in_collection(collection)
        except ValueError:
            click.echo(f"Error: Collection '{collection}' not found.")
            raise SystemExit(1)
    elif search:
        papers = search_papers(search)
    else:
        # Interactive: list collections and let user pick
        try:
            collections = get_collections()
        except FileNotFoundError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)

        if not collections:
            click.echo("No collections found in Zotero library.")
            raise SystemExit(1)

        click.echo("Zotero collections:\n")
        for i, name in enumerate(collections, 1):
            click.echo(f"  {i}. {name}")

        choice = click.prompt("\nSelect a collection", type=click.IntRange(1, len(collections)))
        collection = collections[choice - 1]
        papers = get_papers_in_collection(collection)

    # Step 2: Filter to papers with PDFs
    papers_with_pdf = [p for p in papers if p.pdf_path]

    if not papers:
        click.echo("No papers found.")
        raise SystemExit(1)

    if not papers_with_pdf:
        click.echo(f"Found {len(papers)} paper(s), but none have PDFs attached.")
        raise SystemExit(1)

    # Step 3: Select a paper
    if len(papers_with_pdf) == 1:
        paper = papers_with_pdf[0]
        click.echo(f"\nSelected: {paper.title}")
    else:
        click.echo(f"\nPapers with PDFs ({len(papers_with_pdf)}/{len(papers)}):\n")
        for i, p in enumerate(papers_with_pdf, 1):
            click.echo(f"  {i}. {p.title}")

        choice = click.prompt("\nSelect a paper", type=click.IntRange(1, len(papers_with_pdf)))
        paper = papers_with_pdf[choice - 1]

    # Step 4: Process paper (PDF → cleaned markdown)
    from hearsay.review import process_paper, slugify

    click.echo(f"\nProcessing: {paper.title}")
    click.echo(f"PDF: {paper.pdf_path}\n")

    result = process_paper(
        pdf_path=paper.pdf_path,
        title=paper.title,
        output_dir=output_dir,
        extract_figures=not no_figures,
        describe_figures=not no_figures,
    )

    paper_dir = result["paper_dir"]
    markdown_path = result["markdown_path"]

    # Step 5: Generate script (and audio unless --no-audio)
    from hearsay.tts import create_podcast, generate_script

    markdown_text = markdown_path.read_text(encoding="utf-8")

    if no_audio:
        click.echo("\n[1/1] Generating narration script...")
        script = generate_script(markdown_text, paper.title)
        script_path = paper_dir / "script.txt"
        script_path.write_text(script)
        click.echo(f"  Script: {len(script):,} characters")
        click.echo(f"  Saved to: {script_path}")
    else:
        podcast = create_podcast(
            paper_markdown=markdown_text,
            title=paper.title,
            output_dir=paper_dir,
        )

    # Step 6: Summary
    click.echo("\n" + "=" * 50)
    click.echo("Done!")
    click.echo(f"  Paper dir:  {paper_dir}")
    click.echo(f"  Markdown:   {markdown_path}")
    if no_audio:
        click.echo(f"  Script:     {script_path}")
    else:
        click.echo(f"  Script:     {podcast['script_path']}")
        click.echo(f"  Audio:      {podcast['audio_path']}")
    click.echo("=" * 50)


if __name__ == "__main__":
    main()
