"""
localGeneration.py – Generate responses using a local LM Studio model.

Starts LM Studio headless via 'lms server start', loads the model,
generates a response, unloads the model, and stops the server.
"""

import subprocess

import lmstudio as lms

MODEL = "lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF"


def start_server() -> None:
    """Start LM Studio headless server."""
    subprocess.Popen(["lms", "server", "start"])


def stop_server() -> None:
    """Unload models and stop the server."""
    subprocess.run(["lms", "unload", "--all"], capture_output=True, timeout=60)
    subprocess.run(["lms", "server", "stop"], capture_output=True, timeout=60)


def generate(system_prompt: str, user_prompt: str) -> str:
    """Load model, generate a response, and return it."""
    with lms.Client() as client:
        model = client.llm.model(MODEL)
        chat = lms.Chat(system_prompt)
        chat.add_user_message(user_prompt)
        result = model.respond(chat, config={"temperature": 0.7})
        return str(result)
