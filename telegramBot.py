"""
telegramBot.py – Telegram interface for the contextual retrieval pipeline.

Listens for messages, runs retrieval + generation, and sends back
a generated answer based on the most relevant chunks.
"""

import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from agent import run_agent

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "Send me a question and I'll answer based on my knowledge base."
    )


async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any text message as a retrieval query."""
    query = update.message.text
    await update.message.reply_text("Thinking...")

    answer = run_agent(query)
    await update.message.reply_text(answer)


def main() -> None:
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
