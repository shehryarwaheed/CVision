"""Shared text-cleaning helpers used by clean.py and features.py."""
import hashlib
import html
import re
import unicodedata

from config import SKILLS

_TAG_RE = re.compile(r"<[^>]+>")
_BREAK_RE = re.compile(r"<\s*br\s*/?>|</p>|</li>|</div>", re.I)
_SPECIAL_RE = re.compile(r"[\*\u2022\u25cf\u25aa\u25a0\u2605\u26ab\u2023\u25e6\u00b7\^~|\u2013\u2014]")
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9+#\s]")


def strip_html(text: str) -> str:
    """Remove HTML tags/entities that leaked in from web scraping."""
    text = html.unescape(str(text))
    text = _BREAK_RE.sub(" ", text)
    return _TAG_RE.sub(" ", text)


def clean_text(text: str) -> str:
    """Light cleaning for display: unicode-normalise, drop decorative symbols, collapse spaces."""
    text = unicodedata.normalize("NFKC", strip_html(text))
    text = _SPECIAL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    """Aggressive normalisation for NLP: lowercase, keep letters/digits/+/#."""
    text = _NON_ALNUM_RE.sub(" ", clean_text(text).lower())
    return _WS_RE.sub(" ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(str(text).split())


_SKILL_PATTERN = re.compile(
    r"(?<![a-z0-9+#])("
    + "|".join(re.escape(s) for s in sorted(set(SKILLS), key=len, reverse=True))
    + r")(?![a-z0-9+#])"
)


def extract_skills(normalized_text: str) -> set:
    """Return the set of vocabulary skills that appear in already-normalised text."""
    return set(_SKILL_PATTERN.findall(normalized_text))
