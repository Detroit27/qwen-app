"""
Terminal chatbot — class project.

A small command-line chat app that runs a local Hugging Face model
(Qwen/Qwen3.5-0.8B) fully on CPU. It keeps the whole conversation in
memory so the bot remembers context, and validates all data with
Pydantic v2 models.

Setup:
    pip install -U transformers torch pydantic accelerate
    (transformers MUST be the latest version — the model is from 2026
    and older transformers fails with a KeyError about the model type.)

NOTE: the FIRST run downloads ~2 GB of model weights from Hugging Face
into your local cache and takes a few minutes. Later runs load straight
from the cache — no account or API key needed.

Want a different model? Just change BotConfig.model_name below — e.g. a
Kazakh model like "issai/LLama-3.1-KazLLM-1.0-8B" if you have stronger
hardware (or call it via an API instead of running locally).

Run:
    python chatbot.py        (type "quit" to exit)
"""

import sys
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from transformers import pipeline
from transformers.utils import logging as hf_logging

# Hide transformers' internal warnings (deprecations etc.) so they don't
# clutter the chat — only real errors get printed.
hf_logging.set_verbosity_error()


class BotConfig(BaseModel):
    """Everything tweakable about the bot lives here, with sane limits."""

    model_name: str = "Qwen/Qwen3.5-0.8B"
    max_new_tokens: int = Field(default=200, ge=10, le=1000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatMessage(BaseModel):
    """One chat turn. Validation rejects empty or huge messages early."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


def load_model(config: BotConfig):
    """Load the model, or exit with a friendly explanation if we can't."""
    print(f"Loading {config.model_name} ... (first run downloads ~2 GB, please wait)")
    try:
        return pipeline("text-generation", model=config.model_name)
    except (OSError, KeyError, ValueError) as err:
        # The three usual suspects: no internet, a typo in the model
        # name, or a transformers version too old to know this model.
        print("\nSorry, the model could not be loaded.")
        print(f"Details: {err}")
        print("Likely causes:")
        print("  - No internet connection (needed for the first download)")
        print("  - Misspelled model name in BotConfig")
        print("  - Outdated transformers: run  pip install -U transformers")
        sys.exit(1)


def get_user_input() -> ChatMessage | None:
    """Read one line from the user; return None if it fails validation."""
    text = input("You: ")
    try:
        return ChatMessage(role="user", content=text.strip())
    except ValidationError:
        # Empty or over-2000-char input — warn and let the loop continue
        # without wasting a (slow) model call.
        print("(!) Message must be 1-2000 characters. Try again.")
        return None


def generate_reply(chat, history: list[ChatMessage], config: BotConfig) -> str:
    """Run the model on the full history so it remembers the conversation."""
    # The model wants plain dicts, so we convert only at this boundary
    # and keep validated ChatMessage objects everywhere else.
    messages = [msg.model_dump() for msg in history]
    result = chat(
        messages,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=True,
    )
    # The pipeline returns the whole conversation; the new reply is last.
    return result[0]["generated_text"][-1]["content"].strip()


def main() -> None:
    config = BotConfig()
    chat = load_model(config)

    # Seed the history so the model knows how to behave.
    history: list[ChatMessage] = [
        ChatMessage(
            role="system",
            content="You are a friendly helpful assistant. Keep answers concise.",
        )
    ]

    print("Chat ready! Type 'quit' to exit.\n")
    while True:
        try:
            user_msg = get_user_input()
            if user_msg is None:
                continue  # invalid input, ask again
            if user_msg.content.lower() == "quit":
                print("Goodbye!")
                break

            history.append(user_msg)
            try:
                reply = generate_reply(chat, history, config)
                # Validate the bot's reply too (an empty/huge reply would
                # poison the history for every later turn).
                bot_msg = ChatMessage(role="assistant", content=reply[:2000])
            except Exception as err:
                # Don't kill the whole session over one bad generation —
                # drop the failed turn and keep chatting.
                print(f"(!) Generation failed ({err}). Try again.")
                history.pop()
                continue

            history.append(bot_msg)
            print(f"Bot: {bot_msg.content}\n")
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C / Ctrl+Z should exit politely, not with a traceback.
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
