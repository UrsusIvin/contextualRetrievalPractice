import os
import unicodedata
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

CLIENT = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"

CONTEXT_PROMPT = (
    "<document>\n{document}\n</document>\n"
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context (2-3 sentences) to situate this chunk "
    "within the overall document for the purposes of improving search retrieval of "
    "the chunk. Answer only with the succinct context and nothing else."
)


def to_ascii(text: str) -> str:
    """Transliterate Unicode to nearest ASCII equivalent."""
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", errors="ignore").decode("ascii")


def load_chunks(path: Path) -> list[str]:
    """Parse chunks.txt back into a list of chunk strings."""
    text = path.read_text(encoding="utf-8")
    chunks = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("--- Chunk "):
            if current:
                chunks.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return [c for c in chunks if c]


def generate_context(full_document: str, chunk: str) -> str:
    """Call Claude API to generate a context for a chunk."""
    prompt = CONTEXT_PROMPT.format(
        document=to_ascii(full_document),
        chunk=to_ascii(chunk),
    )
    message = CLIENT.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def main() -> None:
    chunks_path = Path("chunks.txt")
    if not chunks_path.exists():
        print("chunks.txt not found. Run dataPreprocessing.py first.")
        return

    raw_path = Path("raw_text.txt")
    if not raw_path.exists():
        print("raw_text.txt not found. Run getTextFromURL.py first.")
        return

    full_document = raw_path.read_text(encoding="utf-8")
    chunks = load_chunks(chunks_path)

    print(f"Generating context for {len(chunks)} chunks...")

    contextualized = []
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}/{len(chunks)}...", end=" ", flush=True)
        context = generate_context(full_document, chunk)
        contextualized.append(f"CONTEXT: {context}\n\n{chunk}")
        print("done")

    output_path = Path("contextualized_chunks.txt")
    with output_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(contextualized):
            f.write(f"--- Chunk {i + 1} ---\n{chunk}\n\n")

    print(f"\nSaved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
