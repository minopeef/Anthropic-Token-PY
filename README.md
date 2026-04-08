# Claude 3 API–based tokenizer (unofficial)

This repository is **not** an official Anthropic product. It reverse-engineers approximate token boundaries by asking a Claude model to copy text and recording each streaming `content_block_delta` as one segment.

Authors: Javier Rando and Florian Tramèr. Licensed under the terms in `LICENSE` (MIT).

If you use this for security research, follow Anthropic’s own responsible disclosure and terms of service; do not rely on this README for legal or policy guidance.

**Operational note.** Streaming behavior and API details can change. The approach here depends on the Messages API streaming surface exposing text in delta-sized chunks that align with tokenization. If that behavior diverges, results may no longer match the model’s true tokenizer.

## What it does

Anthropic does not ship a public tokenizer for Claude 3 the way some other providers do. This tool approximates tokenization by:

1. Sending a system prompt that instructs the model to copy only the text inside `<tocopy>...</tocopy>` markers.
2. Streaming the response and appending each `event.delta.text` from `content_block_delta` events into a list.
3. Treating that list as a sequence of “tokens” (string pieces as delivered by the stream).

This is slow and consumes API quota, but it can be useful for experiments where rough token boundaries matter more than efficiency.

## Requirements

- Python 3 with `asyncio`
- Install dependencies used by the tokenizer:

```bash
pip install anthropic tqdm
```

The script `scripts/test_tokenization.py` uses the same dependencies as the tokenizer (`anthropic`, `tqdm`) and issues several API calls per test string for cross-model checks.

## Configuration

Set your API key in the environment (the Anthropic client reads `ANTHROPIC_API_KEY`):

```bash
export ANTHROPIC_API_KEY=your_api_key
```

On Windows (cmd), use `set ANTHROPIC_API_KEY=your_api_key`. Running the tokenizer **uses billed tokens** on your account.

## Usage

### Single string

```bash
python src/anthropic_tokenizer.py --text "Security by obscurity is not a good idea."
```

Example output shape:

```
Tokens: ['Security', ' by', ' obsc', 'urity', ' is', ' not', ' a', ' good', ' idea', '.']
Number of text tokens: 10
Total tokens usage (as of API): 13
```

The API-reported output token count is often a few higher than the number of streamed segments (for example, segments plus a small fixed overhead). Treat `usage` fields as authoritative for billing, not as an exact count of the list length.

If the concatenation of streamed segments does not equal the input `--text`, the program raises an exception.

### Batch (JSONL)

Provide a JSONL file where each line is a JSON_object with at least a `text` field. Other fields are preserved.

```bash
python src/anthropic_tokenizer.py --file to_tokenize.jsonl
```

This writes `to_tokenize_tokenized.jsonl` (same basename, `_tokenized` before `.jsonl`) with:

- `tokens` — list of string segments from the stream
- `number_of_tokens` — `len(tokens)`
- `api_total_tokens_usage` — value from the API usage on the final message delta
- `tokenization_correct` — whether `"".join(tokens) == entry["text"]`

Lines that fail during tokenization are printed to stderr; successful rows are still written to the output file.

### Model selection

Default model handle: `claude-3-haiku-20240307`. Override with:

```bash
python src/anthropic_tokenizer.py --text "Hello" --model claude-3-opus-20240229
```

Use a valid model id from Anthropic’s current documentation for your API version.

### Vocabulary log

By default, every run appends each observed segment to `anthropic_vocab.jsonl` (one JSON object per line: `{"token": "..."}`). To turn this off:

```bash
python src/anthropic_tokenizer.py --text "Hello" --disable-vocab
```

### Deduplicate vocabulary

Re-write `anthropic_vocab.jsonl` so each token string appears only once:

```bash
python src/consolidate_vocabulary.py
```

Optional:

```bash
python src/consolidate_vocabulary.py --vocab_file path/to/custom_vocab.jsonl
```

## Repository layout

- `src/anthropic_tokenizer.py` — CLI and streaming tokenization logic
- `src/consolidate_vocabulary.py` — deduplicate `anthropic_vocab.jsonl`
- `scripts/test_tokenization.py` — informal comparisons across Claude 3 model handles (optional; multiple requests per case)
- Sample JSONL inputs may be present at the repo root for quick tests

## Limitations

- **Fidelity.** Stream deltas are an approximation of the true tokenizer. Different models may disagree on edge cases (mixed ASCII and Unicode is called out in comments inside `scripts/test_tokenization.py`).
- **Trailing spaces and newlines.** The system prompt asks the model to preserve them; some inputs may still fail round-trip checks.
- **Newlines and deltas.** The API may bundle a newline with the following characters in one delta; deducing exact token boundaries around line breaks can be ambiguous.
- **Cost and rate limits.** Every string is a full API call with streaming; large batches should be paced accordingly.

## Motivating sanity check (text only)

To see that limiting `max_tokens` on a *different* API call can expose first-token boundaries, you can compare with a public tokenizer on a string unlikely to be one token (for example `asdfasdfasdf`), then ask Claude to copy that string with `max_tokens` set to 1 and 2 in separate experiments and compare prefixes. This repository’s main script uses full streaming of a copy task instead of relying only on `max_tokens`, but the same idea motivates why API behavior relates to tokenization.
