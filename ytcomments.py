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
import os
import re
import sys
import urllib.parse
import shutil
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from youtube_comment_downloader.downloader import (
    YOUTUBE_CONSENT_URL,
    YT_HIDDEN_INPUT_RE,
    YT_INITIAL_DATA_RE,
)

COMMENT_ID_KEYS = ("comment_id", "cid", "id")
AUTHOR_KEYS = ("author", "author_text", "username", "user", "channel")
TEXT_KEYS = ("text", "content", "body", "snippet", "original_text")
TIME_KEYS = ("time", "published_time", "published_at", "date", "timestamp_text")
LIKE_KEYS = ("votes", "like_count", "likes", "likeCount", "favorite_count")
HEART_KEYS = ("heart", "is_hearted", "hearted")
TIME_PARSED_KEYS = ("time_parsed", "timeParsed", "published_at_unix", "timestamp")


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


def _parse_hearted(value: Any) -> bool:
    """Best-effort parse of creator-hearted flag from various payload shapes."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "hearted"}:
            return True
        if v in {"0", "false", "no", "n"}:
            return False
        # Non-empty string: treat as True only if it looks affirmative.
        return False
    if isinstance(value, dict):
        # Some payloads wrap this in a dict. Any explicit truthy flag wins.
        for k in ("hearted", "is_hearted", "heart"):
            if k in value:
                return _parse_hearted(value.get(k))
        return True  # dict present and no explicit flag: assume heart info exists
    if isinstance(value, list):
        return any(_parse_hearted(v) for v in value)
    return False


def _parse_unix_timestamp(value: Optional[str]) -> Optional[int]:
    """Convert a numeric timestamp-ish field into an int unix timestamp (seconds)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return None
    # Keep digits only (tolerates commas or stray chars)
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except (TypeError, ValueError):
        return None


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

    # Prefer structural parent_id passed down from nesting, but fall back to parsing
    # it from comment_id when replies are emitted as separate rows.
    if parent_id is None:
        derived_parent = derive_parent_id_from_comment_id(comment_id)
        if derived_parent:
            parent_id = derived_parent

    author = _first_non_empty(raw_comment, AUTHOR_KEYS, default="Unknown") or "Unknown"
    text = _first_non_empty(raw_comment, TEXT_KEYS, default="") or ""
    published_at = _first_non_empty(raw_comment, TIME_KEYS, default="") or ""
    published_at_unix = _parse_unix_timestamp(_first_non_empty(raw_comment, TIME_PARSED_KEYS))
    is_hearted = _parse_hearted(_first_non_empty(raw_comment, HEART_KEYS))
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
        "published_at_unix": published_at_unix,
        "is_hearted": is_hearted,
        "replies": replies,
    }


def download_comment_threads(url: str) -> Tuple[Optional[str], Optional[int], List[Dict[str, Any]]]:
    """Fetch video metadata and all top-level comments (with replies) for the URL."""
    downloader = YoutubeCommentDownloader()
    title, comment_count = fetch_video_metadata(downloader, url)
    try:
        comment_iter = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)
    except Exception as exc:  # pragma: no cover - library/network errors
        raise RuntimeError(f"Failed to fetch comments: {exc}") from exc

    threads: List[Dict[str, Any]] = []
    progress = tqdm(
        comment_iter,
        total=comment_count,
        unit=" comments",
        desc="Downloading",
        dynamic_ncols=True,
    )
    for raw_comment in progress:
        threads.append(normalize_comment(raw_comment))
    progress.close()
    return title, comment_count, threads


def iter_flatten_comments(comments: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """Yield flattened comment rows (including replies) one by one."""

    def _walk(comment: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        yield {
            "comment_id": comment.get("comment_id", ""),
            "parent_id": comment.get("parent_id") or "",
            "author": comment.get("author", ""),
            "comment_text": comment.get("comment_text", ""),
            "like_count": comment.get("like_count", 0),
            "published_at": comment.get("published_at", ""),
            "published_at_unix": comment.get("published_at_unix"),
            "is_hearted": bool(comment.get("is_hearted", False)),
        }
        for reply in comment.get("replies", []):
            yield from _walk(reply)

    for comment in comments:
        yield from _walk(comment)


TITLE_PATTERN = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
TITLE_SUFFIX = " - YouTube"


INVALID_FS_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def fetch_video_metadata(
    downloader: YoutubeCommentDownloader, url: str
) -> Tuple[Optional[str], Optional[int]]:
    """Return the video title and reported comment count (if available)."""
    try:
        response = downloader.session.get(url)
    except requests.RequestException:
        return None, None

    if "consent" in str(response.url):
        params = dict(re.findall(YT_HIDDEN_INPUT_RE, response.text))
        params.update({"continue": url, "set_eom": False, "set_ytc": True, "set_apyt": True})
        try:
            response = downloader.session.post(YOUTUBE_CONSENT_URL, params=params)
        except requests.RequestException:
            return None, None

    html = response.text
    match = TITLE_PATTERN.search(html)
    title: Optional[str] = None
    if match:
        title = unescape(match.group(1)).strip()
        if title.lower().endswith(TITLE_SUFFIX.lower()):
            title = title[: -len(TITLE_SUFFIX)].rstrip()
        if not title:
            title = None

    comment_count: Optional[int] = None
    initial_data_raw = downloader.regex_search(html, YT_INITIAL_DATA_RE, default="")
    if initial_data_raw:
        try:
            initial_data = json.loads(initial_data_raw)
        except json.JSONDecodeError:
            initial_data = {}
        if isinstance(initial_data, dict):  # defensive
            for count_text in downloader.search_dict(initial_data, "countText"):
                text = None
                if isinstance(count_text, dict):
                    if "simpleText" in count_text:
                        text = count_text.get("simpleText")
                    elif "runs" in count_text:
                        text = "".join(part.get("text", "") for part in count_text.get("runs", []))
                if text and "comment" in text.lower():
                    digits = "".join(ch for ch in text if ch.isdigit())
                    if digits:
                        comment_count = int(digits)
                        break

    return title, comment_count


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


def derive_parent_id_from_comment_id(comment_id: str) -> Optional[str]:
    """Best-effort derive parent_id when the downloader doesn't provide it.

    Some `youtube-comment-downloader` outputs represent reply IDs as:
        <parentCommentId>.<replyId>

    In those cases, `parent_id` may be missing even though the row is a reply.
    """
    if not comment_id:
        return None
    if "." not in comment_id:
        return None
    parent, _reply = comment_id.split(".", 1)
    return parent or None


def interactive_search(flat_rows: List[Dict[str, Any]], output_dir: Path, total_comments: int) -> None:
    """Allow the user to run additional searches with results displayed in a pager."""

    if not flat_rows:
        return

    print("\nEnter additional searches (press Enter to exit).")
    while True:
        try:
            query = input(
                "New search (use quotes for phrases, parentheses allowed, blank to exit): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting search pager.")
            return

        if not query:
            print("Exiting search pager.")
            return

        try:
            matches = keyword_search(flat_rows, query)
        except ValueError as err:
            print(f"Keyword search error: {err}")
            continue

        lines = format_matches(matches)
        summary = [
            "",
            f"Total comments available: {total_comments}",
            f"Comments matching keywords: {len(matches)}",
        ]
        display_lines_paged(lines + summary)

        log_lines = [
            "YouTube Comment Downloader - follow-up search",
            f"Search query: {query}",
            f"Total comments available: {total_comments}",
            f"Comments matching keywords: {len(matches)}",
            "",
            *lines,
        ]
        slug = sanitize_for_filename(query)

        # Human-readable log
        log_path = output_dir / f"search_{slug}.txt"
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"Saved search log to {log_path}")

        # Machine-readable CSV of matched rows
        results_csv_path = output_dir / f"search_{slug}.csv"
        save_search_results_csv(matches, results_csv_path)
        print(f"Saved search results CSV to {results_csv_path}")


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
    fieldnames = [
        "comment_id",
        "parent_id",
        "author",
        "comment_text",
        "like_count",
        "published_at",
        "published_at_unix",
        "is_hearted",
    ]
    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_search_results_csv(matches: List[Dict[str, Any]], path: Path) -> None:
    """Write keyword-matched comment rows to a CSV file."""
    fieldnames = [
        "comment_id",
        "parent_id",
        "author",
        "comment_text",
        "like_count",
        "published_at",
        "published_at_unix",
        "is_hearted",
    ]
    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in matches:
            writer.writerow({key: row.get(key) for key in fieldnames})


def format_matches(matches: List[Dict[str, Any]]) -> List[str]:
    """Return formatted lines for matched comments."""
    if not matches:
        return ["No comments matched the provided keywords."]

    lines = ["Matching comments:"]
    for match in matches:
        snippet = match.get("comment_text", "").strip().replace("\n", " ")
        author = match.get("author") or "Unknown"
        published = match.get("published_at") or "Unknown date"
        lines.append(f"- {author} [{published}]: {snippet}")
        lines.append("")
    return lines


def print_matches(matches: List[Dict[str, Any]], collector: Optional[List[str]] = None) -> None:
    """Display keyword matches with author and the full comment text."""
    lines = format_matches(matches)
    if collector is not None:
        collector.append("")
        collector.extend(lines)

    print()
    display_lines_paged(lines)


def _determine_page_size() -> int:
    size = shutil.get_terminal_size(fallback=(80, 24))
    env_override = os.getenv("YT_PAGER_LINES")
    if env_override:
        try:
            value = int(env_override)
            if value > 0:
                return value
        except ValueError:
            pass
    usable_height = max(size.lines - 4, 5)
    return min(usable_height, 40)


def display_lines_paged(lines: List[str]) -> None:
    """Display lines with a prompt-based pager that adapts to terminal height."""
    if not lines:
        return

    bold_blue = "\033[1m\033[34m"
    reset = "\033[0m"
    index = 0
    total = len(lines)
    page = 1

    while index < total:
        page_size = _determine_page_size()
        if page_size <= 0:
            page_size = 5

        chunk = lines[index : index + page_size]
        header = f"{bold_blue}-- Lines {index + 1}-{index + len(chunk)} of {total} (page {page}) --{reset}"
        print()
        print(header)
        for line in chunk:
            print(line)
        index += len(chunk)

        if index >= total:
            break

        try:
            response = input(
                "--More-- (Enter next lines, b back, q stop) "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if response.startswith("q"):
            break
        if response.startswith("b"):
            index = max(0, index - len(chunk) * 2)
            page = max(1, page - 1)
            continue

        page += 1


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
        video_title, reported_comment_count, comment_threads = download_comment_threads(url)
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
    if reported_comment_count is not None:
        report_lines.append(f"Reported comment count: {reported_comment_count}")

    safe_dir_name = sanitize_directory_name(video_title) if video_title else None
    if not safe_dir_name:
        fallback_id = extract_video_id(url)
        safe_dir_name = sanitize_directory_name(f"video_{fallback_id}") if fallback_id else "video"

    base_output_dir = Path("/Users/jaysair/Documents/Youtube Comments")
    output_dir = base_output_dir / safe_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines.append(f"Output directory: {output_dir}")

    json_path = output_dir / "comments.json"
    csv_path = output_dir / "comments.csv"

    flat_rows: List[Dict[str, Any]] = []
    total_known = reported_comment_count if reported_comment_count and reported_comment_count > 0 else None
    processed = 0
    with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        fieldnames = [
            "comment_id",
            "parent_id",
            "author",
            "comment_text",
            "like_count",
            "published_at",
            "published_at_unix",
            "is_hearted",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        saver = tqdm(
            iter_flatten_comments(comment_threads),
            total=total_known,
            unit=" comments",
            desc="Saving",
            dynamic_ncols=True,
            leave=False,
        )
        for row in saver:
            flat_rows.append(row)
            writer.writerow(row)
            processed += 1
        saver.close()

    progress_summary = (
        f"Processed {processed}/{total_known} comments."
        if total_known
        else f"Processed {processed} comments."
    )
    print(progress_summary)
    report_lines.append(progress_summary)

    save_json(comment_threads, json_path)
    save_line = f"Saved {processed} comments to {json_path} and {csv_path}."
    print(save_line)
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
        # Write search artifacts immediately (before paging output).
        lines = format_matches(matches)
        slug = sanitize_for_filename(query)

        # Human-readable log (search-focused)
        log_lines = [
            "YouTube Comment Downloader - search",
            f"Video URL: {url}",
            title_line,
            f"Search query: {query}",
            f"Total comments downloaded: {processed}",
            f"Comments matching keywords: {len(matches)}",
            "",
            *lines,
        ]
        log_path = output_dir / f"search_{slug}.txt"
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"Saved search log to {log_path}")

        # Machine-readable CSV of matched rows
        results_csv_path = output_dir / f"search_{slug}.csv"
        save_search_results_csv(matches, results_csv_path)
        print(f"Saved search results CSV to {results_csv_path}")

        # Now display results
        print_matches(matches, collector=report_lines)

    total_line = f"Total comments downloaded: {processed}"
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


    interactive_search(flat_rows, output_dir, processed)


if __name__ == "__main__":
    main()
