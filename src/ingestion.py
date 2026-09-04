from pathlib import Path
import re

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "digital_health_handbook.pdf"
)


def clean_text(text: str) -> str:
    """Lightly normalize extracted PDF text while preserving structure."""

    text = text.replace("\xa0", " ")

    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    cleaned_lines = []
    previous_blank = False

    for line in lines:

        if line:
            cleaned_lines.append(line)
            previous_blank = False

        elif not previous_blank:
            cleaned_lines.append("")
            previous_blank = True

    text = "\n".join(cleaned_lines)

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


def extract_pages_from_pdf(
    pdf_path: Path
) -> list[dict]:

    """Extract text from each PDF page while preserving metadata."""

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            text = clean_text(text)

            pages.append(
                {
                    "text": text,
                    "page": page_number,
                    "source": pdf_path.name,
                }
            )

    return pages