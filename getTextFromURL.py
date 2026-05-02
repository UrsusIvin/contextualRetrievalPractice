from html.parser import HTMLParser
from urllib.error import HTTPError, URLError

import urllib.request
from pathlib import Path


class TextExtractor(HTMLParser):
    _skip_tags = {"script", "style", "head", "meta", "link"}

    def __init__(self):
        super().__init__()
        self._skip = False
        self.text_parts = []

    def handle_starttag(self, tag, *_):
        if tag in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self.text_parts.append(stripped)


def fetch_html(url: str) -> str:
    if not url.startswith("https://"):
        url = "https://" + url
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def extract_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return "\n".join(parser.text_parts)


def main() -> None:
    url = input("Enter URL: ").strip()
    if not url:
        print("No URL provided.")
        return

    try:
        html = fetch_html(url)
        text = extract_text(html)
        output_path = Path("output.md")
        output_path.write_text(f"# Extracted Text\n\n{text}", encoding="utf-8")
        print(f"Saved to {output_path.resolve()}")
    except HTTPError as e:
        print(f"HTTP error: {e.code} {e.reason}")
    except URLError as e:
        print(f"URL error: {e.reason}")
    except Exception as e:
        print(f"Failed to fetch HTML: {e}")


if __name__ == "__main__":
    main()