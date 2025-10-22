#!/usr/bin/env python3
"""
Download all public comments from a YouTube video, save them locally, and
search them for keywords (supports quoted phrases plus AND/OR with parentheses).

Requires:
    pip install youtube-comment-downloader
"""

from __future__ import annotations

import csv
import json
import re
import sys
import urllib.parse
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from youtube_comment_downloader.downloader import YOUTUBE_CONSENT_URL, YT_HIDDEN_INPUT_RE

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


def download_comment_threads(url: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Fetch the video title and top-level comments (with replies) for the supplied URL."""
    downloader = YoutubeCommentDownloader()
    title = fetch_video_title(downloader, url)
    try:
        comment_iter = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)
    except Exception as exc:  # pragma: no cover - library/network errors
        raise RuntimeError(f"Failed to fetch comments: {exc}") from exc

    threads: List[Dict[str, Any]] = []
    for raw_comment in comment_iter:
        threads.append(normalize_comment(raw_comment))
    return title, threads


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


TITLE_PATTERN = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
TITLE_SUFFIX = " - YouTube"


INVALID_FS_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def fetch_video_title(downloader: YoutubeCommentDownloader, url: str) -> Optional[str]:
    """Fetch and clean the page title for the given YouTube URL."""
    try:
        response = downloader.session.get(url)
    except requests.RequestException:
        return None

    if "consent" in str(response.url):
        params = dict(re.findall(YT_HIDDEN_INPUT_RE, response.text))
        params.update({"continue": url, "set_eom": False, "set_ytc": True, "set_apyt": True})
        try:
            response = downloader.session.post(YOUTUBE_CONSENT_URL, params=params)
        except requests.RequestException:
            return None

    html = response.text
    match = TITLE_PATTERN.search(html)
    if not match:
        return None

    title = unescape(match.group(1)).strip()
    if title.lower().endswith(TITLE_SUFFIX.lower()):
        title = title[: -len(TITLE_SUFFIX)].rstrip()
    return title or None


def sanitize_directory_name(name: str) -> str:
    """Return a filesystem-friendly directory name derived from the provided string."""
    sanitized = INVALID_FS_CHARS.sub("_", name).strip("._ ")
    return sanitized[:80] or "video"


def sanitize_for_filename(name: str, default: str = "search") -> str:
    """Return a filesystem-safe slug suitable for filenames."""
    sanitized = INVALID_FS_CHARS.sub("_", name).strip("._ ")
    return sanitized[:80] or default


def extract_video_id(url: str) -> Optional[str]:
    """Extract the YouTube video ID from a variety of common URL formats."""
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname in {"youtu.be"}:
        video_id = parsed.path.lstrip("/")
        return video_id or None
    if parsed.hostname and "youtube" in parsed.hostname:
        query = urllib.parse.parse_qs(parsed.query)
        video_id = query.get("v", [""])[0]
        if video_id:
            return video_id
        if parsed.path.startswith("/shorts/"):
            segments = parsed.path.split("/", 2)
            if len(segments) >= 3 and segments[2]:
                return segments[2]
    return None


def _tokenize(query: str) -> List[tuple[str, bool]]:
    """Split the query into tokens, keeping quoted phrases intact."""
    tokens: List[tuple[str, bool]] = []
    i = 0
    length = len(query)
    while i < length:
        ch = query[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append((ch, False))
            i += 1
            continue
        if ch == '"':
            i += 1
            start = i
            while i < length and query[i] != '"':
                i += 1
            if i >= length:
                raise ValueError("Unterminated quote in expression.")
            tokens.append((query[start:i], True))
            i += 1  # Skip closing quote
            continue
        start = i
        while i < length and not query[i].isspace() and query[i] not in '()"':
            i += 1
        tokens.append((query[start:i], False))
    return tokens


class _Node:
    def evaluate(self, row_text: List[str], cache: Dict[str, Set[int]]) -> Set[int]:
        raise NotImplementedError


class _TermNode(_Node):
    def __init__(self, term: str) -> None:
        self.term = term

    def evaluate(self, row_text: List[str], cache: Dict[str, Set[int]]) -> Set[int]:
        if self.term not in cache:
            cache[self.term] = {idx for idx, text in enumerate(row_text) if self.term in text}
        return cache[self.term]


class _BinaryNode(_Node):
    def __init__(self, operator: str, left: _Node, right: _Node) -> None:
        self.operator = operator
        self.left = left
        self.right = right

    def evaluate(self, row_text: List[str], cache: Dict[str, Set[int]]) -> Set[int]:
        left_set = self.left.evaluate(row_text, cache)
        right_set = self.right.evaluate(row_text, cache)
        return left_set & right_set if self.operator == "AND" else left_set | right_set


class _BooleanParser:
    def __init__(self, tokens: List[tuple[str, bool]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def parse_expression(self) -> _Node:
        node = self.parse_conjunction()
        while self._match_operator("OR"):
            rhs = self.parse_conjunction()
            node = _BinaryNode("OR", node, rhs)
        return node

    def parse_conjunction(self) -> _Node:
        node = self.parse_factor()
        while True:
            if self._match_operator("AND"):
                rhs = self.parse_factor()
                node = _BinaryNode("AND", node, rhs)
            elif self._next_starts_factor():
                rhs = self.parse_factor()
                node = _BinaryNode("AND", node, rhs)
            else:
                break
        return node

    def parse_factor(self) -> _Node:
        token = self._peek()
        if token is None:
            raise ValueError("Expression cannot end with an operator.")
        text, quoted = token
        if not quoted:
            upper = text.upper()
            if upper == "AND" or upper == "OR":
                raise ValueError("Expression cannot contain consecutive operators.")
            if text == ")":
                raise ValueError("Expression contains unmatched closing parenthesis.")
            if text == "(":
                self._consume()
                node = self.parse_expression()
                if not self._match_paren(")"):
                    raise ValueError("Expression contains unmatched opening parenthesis.")
                return node
        self._consume()
        term = text.lower()
        return _TermNode(term)

    def _match_operator(self, operator: str) -> bool:
        token = self._peek()
        if token and not token[1] and token[0].upper() == operator:
            self._consume()
            return True
        return False

    def _match_paren(self, paren: str) -> bool:
        token = self._peek()
        if token and not token[1] and token[0] == paren:
            self._consume()
            return True
        return False

    def _next_starts_factor(self) -> bool:
        token = self._peek()
        if token is None:
            return False
        text, quoted = token
        if not quoted:
            upper = text.upper()
            if upper in {"AND", "OR"} or text == ")":
                return False
        return True

    def _peek(self) -> Optional[tuple[str, bool]]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume(self) -> None:
        self.pos += 1


def keyword_search(rows: Iterable[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Return comments that satisfy the AND/OR expression built from the query string."""
    tokens = _tokenize(query)
    if not tokens:
        return []
    parser = _BooleanParser(tokens)
    root = parser.parse_expression()
    if parser._peek() is not None:
        raise ValueError("Malformed keyword expression.")

    row_text = [str(row.get("comment_text", "")).lower() for row in rows]
    cache: Dict[str, Set[int]] = {}
    matching_indices = root.evaluate(row_text, cache)
    return [row for idx, row in enumerate(rows) if idx in matching_indices]


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


def print_matches(matches: List[Dict[str, Any]], collector: Optional[List[str]] = None) -> None:
    """Display keyword matches with author and the full comment text."""

    def emit(line: str = "") -> None:
        if collector is not None:
            collector.append(line)
        print(line)

    if not matches:
        emit()
        emit("No comments matched the provided keywords.")
        return

    emit()
    emit("Matching comments:")
    for match in matches:
        snippet = match.get("comment_text", "").strip().replace("\n", " ")
        author = match.get("author") or "Unknown"
        published = match.get("published_at") or "Unknown date"
        emit(f"- {author} [{published}]: {snippet}")


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
        query = input(
            "Enter keywords with AND/OR (use quotes for phrases, parentheses allowed, leave blank to skip search): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nKeyword search skipped.")
        query = ""

    try:
        video_title, comment_threads = download_comment_threads(url)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    report_lines: List[str] = [
        "YouTube Comment Downloader",
        f"Video URL: {url}",
    ]

    if video_title:
        title_line = f"Video title: {video_title}"
        print(f"\n{title_line}")
    else:
        title_line = "Video title: (unknown)"
        print(f"\n{title_line}")
    report_lines.append(title_line)

    safe_dir_name = sanitize_directory_name(video_title) if video_title else None
    if not safe_dir_name:
        fallback_id = extract_video_id(url)
        safe_dir_name = sanitize_directory_name(f"video_{fallback_id}") if fallback_id else "video"

    output_dir = Path.cwd() / safe_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines.append(f"Output directory: {output_dir}")

    flat_rows = flatten_comments(comment_threads)
    json_path = output_dir / "comments.json"
    csv_path = output_dir / "comments.csv"

    save_json(comment_threads, json_path)
    save_csv(flat_rows, csv_path)
    save_line = f"Saved {len(flat_rows)} comments to {json_path} and {csv_path}."
    print(f"\n{save_line}")
    report_lines.append(save_line)

    search_performed = bool(query)
    matches: List[Dict[str, Any]] = []
    if search_performed:
        report_lines.append(f"Search query: {query}")
        try:
            matches = keyword_search(flat_rows, query)
        except ValueError as err:
            print(f"\nKeyword search error: {err}")
            search_performed = False

    if search_performed:
        print_matches(matches, collector=report_lines)

    total_line = f"Total comments downloaded: {len(flat_rows)}"
    print(f"\n{total_line}")
    report_lines.append("")
    report_lines.append(total_line)
    if search_performed:
        matches_line = f"Comments matching keywords: {len(matches)}"
        print(matches_line)
        report_lines.append(matches_line)
    else:
        print("Keyword search skipped.")
        report_lines.append("Keyword search skipped.")

    if search_performed:
        slug = sanitize_for_filename(query)
        log_path = output_dir / f"search_{slug}.txt"
        log_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Saved search log to {log_path}")


if __name__ == "__main__":
    main()
