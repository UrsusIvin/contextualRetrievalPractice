import re
from pathlib import Path


# --- Suitability checks ---

MIN_WORDS = 50
MIN_ASCII_RATIO = 0.85


def check_suitability(text: str) -> list[str]:
    """Return a list of warnings if text is unsuitable; empty list if all good."""
    warnings = []
    words = text.split()

    if len(words) < MIN_WORDS:
        warnings.append(f"Too short: {len(words)} words (minimum {MIN_WORDS})")

    non_ascii = sum(1 for c in text if ord(c) > 127)
    ascii_ratio = 1 - non_ascii / max(len(text), 1)
    if ascii_ratio < MIN_ASCII_RATIO:
        warnings.append(f"Low ASCII ratio: {ascii_ratio:.0%} — may contain garbled encoding")

    if not any(c.isalpha() for c in text):
        warnings.append("No alphabetic characters found")

    return warnings


_NAV_PATTERNS = re.compile(
    r"^(skip to|log in|sign up|download app|© \d{4}|privacy policy|terms of|"
    r"usage policy|support center|status|careers|pricing|claude for |"
    r"claude code|claude cowork|solutions|resources|company|products|"
    r"models|opus|sonnet|haiku|help and security|consumer health|"
    r"responsible|transparency|availability|get the developer|"
    r"please provide your email|you can unsubscribe)",
    re.IGNORECASE,
)


def _is_nav_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped.split()) <= 4 and _NAV_PATTERNS.match(stripped):
        return True
    return False


def clean_text(text: str) -> str:
    """Remove nav/footer noise and normalize whitespace."""
    lines = text.splitlines()
    cleaned = [line for line in lines if not _is_nav_line(line)]
    return "\n".join(cleaned).strip()


# --- Chunking ---

CHUNK_SIZE = 200      # target words per chunk
CHUNK_OVERLAP = 40   # words of overlap between chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? boundaries."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks that always end and start on sentence boundaries."""
    sentences = _split_sentences(text)
    chunks = []
    start_idx = 0

    while start_idx < len(sentences):
        word_count = 0
        end_idx = start_idx
        # Accumulate sentences until we reach or exceed chunk_size
        while end_idx < len(sentences):
            word_count += len(sentences[end_idx].split())
            end_idx += 1
            if word_count >= chunk_size:
                break

        chunks.append(" ".join(sentences[start_idx:end_idx]))

        if end_idx == len(sentences):
            break

        # Find overlap start: walk back from end_idx until overlap word target is reached
        overlap_words = 0
        overlap_start = end_idx
        while overlap_start > start_idx:
            overlap_words += len(sentences[overlap_start - 1].split())
            overlap_start -= 1
            if overlap_words >= overlap:
                break

        start_idx = overlap_start

    return chunks


# --- Main ---

def main() -> None:
    input_path = Path("raw_text.txt")
    if not input_path.exists():
        print("raw_text.txt not found. Run getTextFromURL.py first.")
        return

    text = input_path.read_text(encoding="utf-8")

    warnings = check_suitability(text)
    if warnings:
        print("Suitability warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("Suitability check passed.")

    text = clean_text(text)
    chunks = chunk_text(text)

    print(f"\nProduced {len(chunks)} chunks from {len(text.split())} words.")

    output_path = Path("chunks.txt")
    with output_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i + 1} ---\n{chunk}\n\n")

    print(f"Saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
