def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """
    Split text into overlapping chunks of approximately chunk_size characters.

    Tries to break on natural boundaries (paragraph > line > sentence > space)
    so chunks don't end mid-word or mid-sentence.

    Args:
        text: the input string
        chunk_size: target max characters per chunk
        overlap: characters from end of one chunk repeated at start of next
                 (preserves context across boundaries)

    Returns:
        list of non-empty stripped chunk strings
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # If not at the end of the text, try to break on a natural boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = chunk.rfind(sep)
                # only break if the boundary is reasonably far, don't make tiny chunks
                if idx > chunk_size * 0.5:
                    chunk = chunk[:idx + len(sep)]
                    end = start + idx + len(sep)
                    break

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        # next chunk starts before the current one ended, to overlap
        start = end - overlap

    return chunks