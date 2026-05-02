# CLAUDE.md

## Project
`contextualRetrievalPractice` — a project for experimenting with contextual retrieval techniques (e.g., RAG).
    `getTextFromURL` - script collecting text from a fetched URL
    `markdownTheWebText` - turns the collected text into a markdown file for my viewing pleasure
    
## Environment
- Python virtual environment: `.contextvenv`
- Activate: `source .contextvenv/Scripts/activate` (Git Bash)
- Validate active venv: `which python` or `import sys; print(sys.executable)`
- VSCode interpreter: set to `.contextvenv\Scripts\python.exe` via `Ctrl+Shift+P` → Python: Select Interpreter

## Project Elements
Separat scripts for separate functions.
1. getTextFromURL - takes a https webpage and scrapes all the text from it, storing it in a .txt file.
2. dataPreprocessing - takes the initial text, checks if it is longer than 50 words, checks how much of it is non-ASCII. This is followed by removal of common navigational element's texts and single word sentences. This is followed by chunking, target 200, target overlap 40. each chunk is close to 200 words, without cutting a sentence off. Each overlap is close to 40 words, without cutting centences off.