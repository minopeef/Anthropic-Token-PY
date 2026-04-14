from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from anthropic import AsyncAnthropic
from tqdm import tqdm


async def get_tokens(
    client: AsyncAnthropic, to_tokenize: str, model: str | None = None
) -> tuple[list[str], int]:
    """
    Stream a copy of ``to_tokenize`` and collect each text delta as one segment.

    Default model is ``claude-3-haiku-20240307``. ``scripts/test_tokenization.py`` shows
    segment lists often match across Claude 3 handles, except for some mixed ASCII/Unicode cases.
    """
    if model is None:
        model = "claude-3-haiku-20240307"
    tokens: list[str] = []
    total_tokens_usage = 0
    async with client.messages.stream(
        max_tokens=1000,
        system=(
            "Copy the text between <tocopy> markers. Include trailing spaces or breaklines."
            " Do not write anything else. One example \nInput: <tocopy>Example"
            " sentence.</tocopy>\nOutput: Example sentence."
        ),
        messages=[
            {
                "role": "user",
                "content": f"<tocopy>{to_tokenize}</tocopy>",
            }
        ],
        model=model,
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                tokens.append(event.delta.text)
            if event.type == "message_delta":
                total_tokens_usage = event.usage.output_tokens

    return tokens, total_tokens_usage


def tokenize_text(
    client: AsyncAnthropic, to_tokenize: str, model: str | None = None
) -> tuple[list[str], int]:
    tokens, total_tokens_usage = asyncio.run(get_tokens(client, to_tokenize, model=model))
    return tokens, total_tokens_usage


def _batch_output_path(input_jsonl: str | Path) -> Path:
    p = Path(input_jsonl)
    return p.with_name(f"{p.stem}_tokenized{p.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="The text to tokenize", required=False, default=None)
    parser.add_argument(
        "--model",
        help="The model to be used for inference. Use a handle from Anthropic docs.",
        required=False,
        default="claude-3-haiku-20240307",
    )
    parser.add_argument(
        "--file",
        help="A JSONL file with several texts to be tokenized",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSONL path for batch mode (default: <input_stem>_tokenized.jsonl next to the input file)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--vocab-file",
        help="Append observed segments to this JSONL (one object per line: {\"token\": \"...\"})",
        required=False,
        default="anthropic_vocab.jsonl",
    )
    parser.add_argument(
        "--disable-vocab",
        help="Do not append to the vocabulary file",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.text and not args.file:
        print("anthropic_tokenizer: error: You must provide either --text or --file.", file=sys.stderr)
        sys.exit(2)

    keep_vocab = not args.disable_vocab
    vocab_path = Path(args.vocab_file)

    client = AsyncAnthropic()

    if args.text:
        tokens, total_tokens_usage = tokenize_text(client, args.text, args.model)
        print("Tokens:", tokens)
        print("Number of text tokens:", len(tokens))
        print("Total tokens usage (as of API):", total_tokens_usage)

        if keep_vocab:
            with vocab_path.open("a", encoding="utf-8") as f:
                for t in tokens:
                    f.write(json.dumps({"token": t}) + "\n")

        if "".join(tokens) != args.text:
            raise RuntimeError(
                "The tokenization resulted in a different string than the original. See below:\n\n"
                "========= Original =========\n{}\n\n\n========= Tokenized =========\n{}".format(
                    args.text, "".join(tokens)
                )
            )

    if args.file:
        input_path = Path(args.file)
        out_path = Path(args.output) if args.output else _batch_output_path(input_path)

        to_tokenize: list[dict] = []
        with input_path.open(encoding="utf-8") as f:
            for line in f:
                to_tokenize.append(json.loads(line))

        for entry in tqdm(to_tokenize):
            try:
                tokens, total_tokens_usage = tokenize_text(client, entry["text"], args.model)
                entry["tokens"] = tokens
                entry["number_of_tokens"] = len(tokens)
                entry["api_total_tokens_usage"] = total_tokens_usage
                entry["tokenization_correct"] = "".join(tokens) == entry["text"]
                entry.pop("tokenization_error", None)
            except Exception as e:
                print(f"Error tokenizing text: {entry['text']}", file=sys.stderr)
                print(e, file=sys.stderr)
                entry["tokenization_correct"] = False
                entry["tokenization_error"] = repr(e)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for entry in to_tokenize:
                f.write(json.dumps(entry) + "\n")

        if keep_vocab:
            batch_tokens: set[str] = set()
            for entry in to_tokenize:
                if "tokens" not in entry:
                    continue
                batch_tokens.update(entry["tokens"])
            with vocab_path.open("a", encoding="utf-8") as f:
                for t in batch_tokens:
                    f.write(json.dumps({"token": t}) + "\n")


if __name__ == "__main__":
    main()
