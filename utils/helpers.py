"""General helper functions."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from typing import Any


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# Sentinel used to protect literal braces while a template is being filled in.
# It is deliberately long and unlikely to appear in real prompt/document text.
_SENTINEL = "__SAFE_FORMAT_SENTINEL_1f9c8b2a__"


def safe_format(template: str, **kwargs: Any) -> str:
    """Format a template without crashing on braces inside the value text.

    ``str.format`` treats any ``{...}`` in the supplied values or in unknown
    template fields as format specifiers and raises ``KeyError``/``ValueError``.
    Document text and user questions routinely contain ``{`` ``}`` (e.g. JSON
    snippets, code, LaTeX, ``{ }``), which can make retrieval/answer prompts
    fail silently.

    This helper behaves like ``str.format`` for the placeholders we actually use
    (``{key}``), unescapes author-written ``{{`` ``}}`` exactly like
    ``str.format``, and leaves any unknown ``{...}``/braces untouched instead of
    raising.
    """
    # 1) Protect braces the prompt author escaped as {{ }} / }} so they are
    #    restored as literal { } in the final output (str.format behavior).
    template = template.replace("{{", _SENTINEL + "L")
    template = template.replace("}}", _SENTINEL + "R")

    # 2) Replace only the named placeholders that were provided.
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in kwargs:
            return str(kwargs[key])
        # Unknown placeholder -> keep it as-is rather than raising.
        return match.group(0)

    template = re.sub(r"\{(\w+)\}", _replace, template)

    # 3) Restore protected literal braces.
    template = template.replace(_SENTINEL + "L", "{")
    template = template.replace(_SENTINEL + "R", "}")
    return template