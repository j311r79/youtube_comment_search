#!/usr/bin/env python3
"""
Download all public comments from a YouTube video, save them locally, and
search them for keywords.

Requires:
    pip install youtube-comment-downloader
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT

COMMENT_ID_KEYS = ("comment_id", "cid", "id")
AUTHOR_KEYS = ("author", "author_text", "username", "user", "channel")
TEXT_KEYS = ("text", "content", "body", "snippet", "original_text")
TIME_KEYS = ("time", "published_time", "published_at", "date", "timestamp_text")
LIKE_KEYS = ("votes", "like_count", "likes", "likeCount", "favorite_count")


def _first_non_empty(
    data: Dict[str, Any],
    keys: Iterable[str],
    default: Optional[str] = None,
) -> Optional[str]:
    """Return the first non-empty value (as a string) for the provided keys."""
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, list):
            value = " ".join(str(item) for item in value if item)
        if isinstance(value, str):
            value = value.strip()
        if value in (None, "", []):
            continue
        if isinstance(value, (int, float)):
            return str(int(value))
        return str(value)
    return default


def _parse_like_count(value: Optional[str]) -> int:
    """Convert the like-count field to an integer, ignoring non-digit characters."""
    if not value:
        return 0
    digits = "".join(ch for ch in value if ch.isdigit())
    if digits:
        return int(digits)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _extract_replies(raw_comment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect any reply dictionaries present on the raw comment payload."""
    replies: List[Dict[str, Any]] = []
    for key in ("replies", "reply_comments", "children", "comments"):
        value = raw_comment.get(key)
        if isinstance(value, list):
            replies.extend(value)
        elif isinstance(value, dict):
            replies.extend(value.values())
    direct_reply = raw_comment.get("reply")
    if isinstance(direct_reply, list):
        replies.extend(direct_reply)
    elif isinstance(direct_reply, dict):
        replies.append(direct_reply)
    return replies


def normalize_comment(raw_comment: Dict[str, Any], parent_id: Optional[str] = None) -> Dict[str, Any]:
    """Map a raw comment from youtube-comment-downloader to a consistent structure."""
    comment_id = _first_non_empty(raw_comment, COMMENT_ID_KEYS, default="") or ""
    author = _first_non_empty(raw_comment, AUTHOR_KEYS, default="Unknown") or "Unknown"
    text = _first_non_empty(raw_comment, TEXT_KEYS, default="") or ""
    published_at = _first_non_empty(raw_comment, TIME_KEYS, default="") or ""
    like_count = _parse_like_count(_first_non_empty(raw_comment, LIKE_KEYS))

    replies = [
        normalize_comment(reply, comment_id if comment_id else parent_id)
        for reply in _extract_replies(raw_comment)
    ]

    return {
        "comment_id": comment_id,
        "parent_id": parent_id,
        "author": author,
        "comment_text": text,
        "like_count": like_count,
        "published_at": published_at,
        "replies": replies,
    }


def download_comment_threads(url: str) -> List[Dict[str, Any]]:
    """Fetch each top-level comment (with replies) for the supplied YouTube URL."""
    downloader = YoutubeCommentDownloader()
    try:
        comment_iter = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)
    except Exception as exc:  # pragma: no cover - library/network errors
        raise RuntimeError(f"Failed to fetch comments: {exc}") from exc

    threads: List[Dict[str, Any]] = []
    for raw_comment in comment_iter:
        threads.append(normalize_comment(raw_comment))
    return threads


def flatten_comments(comments: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten nested comment threads into a list suitable for CSV export."""
    rows: List[Dict[str, Any]] = []

    def _walk(comment: Dict[str, Any]) -> None:
        rows.append(
            {
                "comment_id": comment.get("comment_id", ""),
                "parent_id": comment.get("parent_id") or "",
                "author": comment.get("author", ""),
                "comment_text": comment.get("comment_text", ""),
                "like_count": comment.get("like_count", 0),
                "published_at": comment.get("published_at", ""),
            }
        )
        for reply in comment.get("replies", []):
            _walk(reply)

    for comment in comments:
        _walk(comment)
    return rows


def keyword_search(rows: Iterable[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Return comments containing every keyword (case-insensitive) from the query string."""
    terms = [term.lower() for term in query.split() if term.strip()]
    if not terms:
        return []
    matches: List[Dict[str, Any]] = []
    for row in rows:
        text_lower = row.get("comment_text", "").lower()
        if all(term in text_lower for term in terms):
            matches.append(row)
    return matches


def save_json(comments: List[Dict[str, Any]], path: Path) -> None:
    """Write the nested comment data to JSON."""
    path.write_text(json.dumps(comments, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write the flattened comment data to CSV."""
    fieldnames = ["comment_id", "parent_id", "author", "comment_text", "like_count", "published_at"]
    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_matches(matches: List[Dict[str, Any]]) -> None:
    """Display keyword matches with author and the full comment text."""
    if not matches:
        print("\nNo comments matched the provided keywords.")
        return

    print("\nMatching comments:")
    for match in matches:
        snippet = match.get("comment_text", "").strip().replace("\n", " ")
        author = match.get("author") or "Unknown"
        published = match.get("published_at") or "Unknown date"
        print(f"- {author} [{published}]: {snippet}")


def main() -> None:
    """Drive the download, persistence, keyword search, and summary reporting."""
    print("YouTube Comment Downloader\n")
    try:
        url = input("Enter the YouTube video URL: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo URL provided. Exiting.")
        return

    if not url:
        print("No URL provided. Exiting.")
        return

    try:
        comment_threads = download_comment_threads(url)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    flat_rows = flatten_comments(comment_threads)
    json_path = Path.cwd() / "comments.json"
    csv_path = Path.cwd() / "comments.csv"

    save_json(comment_threads, json_path)
    save_csv(flat_rows, csv_path)
    print(f"\nSaved {len(flat_rows)} comments to {json_path.name} and {csv_path.name}.")

    try:
        query = input("Enter keyword(s) separated by spaces (leave blank to skip search): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nKeyword search skipped.")
        query = ""

    matches = keyword_search(flat_rows, query) if query else []
    if query:
        print_matches(matches)

    print(f"\nTotal comments downloaded: {len(flat_rows)}")
    if query:
        print(f"Comments matching keywords: {len(matches)}")
    else:
        print("Keyword search skipped.")


if __name__ == "__main__":
    main()
