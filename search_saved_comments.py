#!/usr/bin/env python3
"""Search previously downloaded YouTube comments using boolean expressions.

The script assumes it is executed from a directory containing the
``comments.csv`` produced by ``youtube_comment_scraper.py``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

from youtube_comment_scraper import keyword_search, print_matches

COMMENTS_FILENAME = "comments.csv"


def load_flattened_comments(csv_path: Path) -> List[Dict[str, str]]:
    """Load the flattened comments from the provided CSV path."""
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    return rows


def prompt(prompt_text: str) -> str:
    """Safely prompt the user and handle EOF/interrupts."""
    try:
        return input(prompt_text)
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled. Exiting.")
        sys.exit(0)


def main() -> None:
    """Drive the keyword-search workflow for saved comment exports."""
    print("YouTube Comment Search (saved data)\n")

    try:
        csv_path = Path.cwd() / COMMENTS_FILENAME
        if not csv_path.is_file():
            raise FileNotFoundError(f"Expected {COMMENTS_FILENAME} in {Path.cwd()}")
        flat_rows = load_flattened_comments(csv_path)
    except Exception as err:
        print(f"Failed to load {COMMENTS_FILENAME}: {err}", file=sys.stderr)
        sys.exit(1)

    if not flat_rows:
        print(f"No comments found in {COMMENTS_FILENAME}. Exiting.")
        return

    query = prompt(
        "Enter keywords with AND/OR (use quotes for phrases, parentheses allowed, leave blank to skip search): "
    ).strip()

    search_performed = bool(query)
    matches: List[Dict[str, str]] = []
    if search_performed:
        try:
            matches = keyword_search(flat_rows, query)
        except ValueError as err:
            print(f"\nKeyword search error: {err}")
            search_performed = False

    if search_performed:
        print_matches(matches)

    print(f"\nLoaded comments from: {csv_path}")
    print(f"Total comments available: {len(flat_rows)}")
    if search_performed:
        print(f"Comments matching keywords: {len(matches)}")
    else:
        print("Keyword search skipped.")


if __name__ == "__main__":
    main()
