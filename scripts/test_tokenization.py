"""
Compare streaming "tokens" across Claude 3 model handles for a fixed battery of strings.

Run from the repository root:

    python scripts/test_tokenization.py

Requires ANTHROPIC_API_KEY and the anthropic Python SDK (same as the main tokenizer).
"""

from __future__ import annotations

import argparse
import random
import sys
from itertools import zip_longest
from pathlib import Path
import string
from typing import List

from anthropic import AsyncAnthropic

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.anthropic_tokenizer import tokenize_text

MODEL_OPUS = "claude-3-opus-20240229"
MODEL_SONNET = "claude-3-sonnet-20240229"
MODEL_HAIKU = "claude-3-haiku-20240307"

# Empirically, Haiku and Opus often agree; Sonnet can differ on mixed ASCII/Unicode (see cases s4, s5).


def _get_random_unicode(length: int) -> str:
    include_ranges = [
        (0x18B0, 0x18F5),
        (0x1900, 0x194F),
        (0x1950, 0x1974),
        (0x1980, 0x19DF),
        (0x19E0, 0x19FF),
        (0x1A00, 0x1A1F),
        (0x1A20, 0x1AAD),
        (0x1B00, 0x1B7C),
        (0x1B80, 0x1BB9),
        (0x1BC0, 0x1BFF),
        (0x1C00, 0x1C4F),
        (0x1C50, 0x1C7F),
        (0x1CD0, 0x1CF2),
        (0x1D00, 0x1D7F),
    ]
    alphabet = [
        chr(code_point)
        for current_range in include_ranges
        for code_point in range(current_range[0], current_range[1] + 1)
    ]
    return "".join(random.choice(alphabet) for _ in range(length))


def build_test_strings(seed: int | None) -> List[str]:
    """Build the same strings as the original notebook-style script; random parts depend on seed."""
    if seed is not None:
        random.seed(seed)

    s1 = (
        "We hold these truths to be self-evident, that all men are created equal, that they are"
        " endowed, by their Creator, with certain unalienable rights, that among these are life,"
        " liberty, and the pursuit of happiness."
    )
    s2 = """We believe AI will have a vast impact on the world. Anthropic is dedicated to building systems that people can rely on and generating research about the opportunities and risks of AI. We Build Safer Systems."""
    s3 = """Ↄ⊙◵⳺⸬⩋ⷒ⊨⃺ⴾ❭⬫⼀⠹⍓⩁⤋┝⢫⁴⨂☍⭬➖⒛⯩ⓓ⁊⬬✟⮧⇡⍰⪤⚉⣶⃏⍮ⷕ⤽⸍⩽⤍➦➟⊞⁶⟈⬹⠤ⲵ‸◢⤽⤷∺⻊ⓨ➑⊳⍖✚☌‎⾹ⲡ✂⾠⩾ⵟ≑⎃⪆◻╾⶗ⅴ➙ⴗ▌❗◿☎ⴻ▱⍗⃌⹃⹕➨⌚⊝∆⠐ⳋ⍆⫭⃃⤽⻀"""

    s4 = "".join([c for i, j in zip_longest(s1.split(" "), list(s3), fillvalue="") for c in (i, j)])
    s5 = "".join([c for i, j in zip_longest(s2.split(" "), list(s3), fillvalue="") for c in (i, j)])

    s6 = " ".join(
        "".join(random.choice(string.ascii_letters) for _ in range(random.randint(1, 10)))
        for _ in range(20)
    )
    s7 = " ".join(
        "".join(random.choice(string.ascii_letters) for _ in range(random.randint(1, 10)))
        for _ in range(20)
    )

    s8 = "".join(random.choice(string.ascii_letters) for _ in range(100))
    s9 = "".join(random.choice(string.ascii_letters) for _ in range(100))

    u_hi = 0xD7FF
    s10 = "".join(
        (
            random.choice(string.ascii_letters)
            if random.random() < 0.5
            else chr(random.randint(0, u_hi))
        )
        for _ in range(100)
    )
    s11 = "".join(
        (
            random.choice(string.ascii_letters)
            if random.random() < 0.5
            else chr(random.randint(0, u_hi))
        )
        for _ in range(100)
    )

    s12 = "".join(chr(random.choice(range(256))) for _ in range(100))
    s13 = "".join(chr(random.choice(range(256))) for _ in range(100))

    s14 = (
        "Ḽơᶉëᶆ ȋṕšᶙṁ ḍỡḽǭᵳ ʂǐť ӓṁệẗ, ĉṓɲṩḙċťᶒțûɾ ấɖḯƥĭṩčįɳġ ḝłįʈ, șếᶑ ᶁⱺ ẽḭŭŝḿꝋď ṫĕᶆᶈṓɍ ỉñḉīḑȋᵭṵńť ṷŧ"
        " ḹẩḇőꝛế éȶ đꝍꞎôꝛȇ ᵯáꞡᶇā ąⱡîɋṹẵ."
    )

    s15 = "�����������������������������"

    s16 = _get_random_unicode(100)
    s17 = _get_random_unicode(100)

    s18 = "".join(chr(i) for i in [7496, 7387, 7020])

    return [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18]


def _tokens_match(a, b) -> bool:
    return a[0] == b[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model tokenization comparison for Claude 3.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random test strings (s6 through s13 and s16 through s17). Default: nondeterministic.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only run the first N test strings (after building the full list).",
    )
    args = parser.parse_args()

    strings = build_test_strings(args.seed)
    if args.limit is not None:
        strings = strings[: args.limit]

    client = AsyncAnthropic()

    for ix, s in enumerate(strings):
        o1 = tokenize_text(client, s, model=MODEL_OPUS)
        o2 = tokenize_text(client, s, model=MODEL_SONNET)
        o3 = tokenize_text(client, s, model=MODEL_HAIKU)

        good = [_tokens_match(o1, o2), _tokens_match(o2, o3), _tokens_match(o1, o3)]

        for name, token_list in zip(
            ("opus", "sonnet", "haiku"),
            (o1[0], o2[0], o3[0]),
        ):
            merged = "".join(token_list)
            if merged != s:
                print(f"WARN: case {ix} model {name} round-trip mismatch")
                print(f"  expected repr: {s!r}")
                print(f"  got repr:      {merged!r}")
                print(f"  diff positions: {[(p, (a, b)) for p, (a, b) in enumerate(zip(s, merged)) if a != b]}")

        if not all(good):
            print(ix, s)
            print("token lists match across models:", good)
            print(o1, o2, o3, sep="\n")
            print([(p, i, j) for p, (i, j) in enumerate(zip(o1[0], o2[0])) if i != j])
            print([(p, i, j) for p, (i, j) in enumerate(zip(o2[0], o3[0])) if i != j])
            print([(p, i, j) for p, (i, j) in enumerate(zip(o1[0], o3[0])) if i != j])
        else:
            preview = s if len(s) <= 80 else s[:77] + "..."
            print(f"checked {ix}: {preview!r}")


if __name__ == "__main__":
    main()
