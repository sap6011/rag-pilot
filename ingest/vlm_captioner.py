"""
Vision-Language Model captioner for image-heavy PDF pages.

When a PDF page has very little text (mostly a diagram or image), we render
the page to an image and ask a VLM to describe it. The resulting caption is
appended to the page's text so downstream chunking/embedding/retrieval works
normally — the rest of the pipeline doesn't need to know vision was involved.
"""
from pathlib import Path
from pdf2image import convert_from_path
from PIL.Image import Image
import ollama
import tempfile

# Pages with less than this many characters of extracted text are treated as
# image-heavy and sent to the VLM. We can tune this threshold via eval: lower = more captioning
# (slower ingestion, better recall on diagrams).

CAPTION_THRESHOLD_CHARS = 200

VLM_MODEL = "qwen3-vl:4b"

CAPTION_PROMPT = (
    "Describe this slide concisely in 2-3 sentences. "
    "Focus on named concepts, stages, labels, and any text inside diagrams. "
    "Do not describe colors, shapes, or visual styling."
)

def render_pdf_pages(pdf_path: Path, dpi: int = 150) -> list[Image]:
    """Render every page of a PDF to a PIL image. Returns one image per page, in order."""
    return convert_from_path(str(pdf_path), dpi=dpi)

def needs_captioning(text: str) -> bool:
    """True if the page's text layer is sparse enough to warrant a VLM caption."""
    return len(text.strip()) < CAPTION_THRESHOLD_CHARS

def caption_image(image: Image) -> str:
    """Send a PIL image to the VLM and return its caption."""
    # ollama expects an image path, not a PIL object, so we write to a temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.save(tmp_path)
        response = ollama.chat(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": CAPTION_PROMPT,
                "images": [tmp_path],
            }],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"  VLM captioning failed: {e}")
        return ""
    finally:
        Path(tmp_path).unlink(missing_ok=True)