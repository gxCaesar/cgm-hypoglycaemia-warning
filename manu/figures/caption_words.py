"""Count a figure's caption words by reading the manuscript, not a copy of it.

The height a journal allows is a function of the caption's length, so the geometry
check needs the caption that is actually in the paper. Passing that count as a literal
makes it a restated number, and a restated number drifts from its source the first time
someone edits the caption without touching the builder. The count is therefore derived
here, on every build.
"""

from __future__ import annotations

import re
from pathlib import Path

_MANU = Path(__file__).resolve().parent.parent
_SECTIONS = _MANU / "sections"
# A figure may live in the supplement. Scanning only the main text made the build
# script die on the first figure the moment one was moved, and because the script
# is set -e every later figure then went unrebuilt while looking fine.
_DOCUMENTS = sorted(_SECTIONS.glob("*.tex")) + [_MANU / "supplement.tex"]
_FIGURE = re.compile(
    r"\\includegraphics[^{]*\{figures/(?P<stem>fig_[a-z0-9_]+)\.pdf\}"
    r"(?P<body>.*?)\\end\{figure",
    re.S,
)
_CAPTION = re.compile(r"\\caption\{(?P<text>.*)", re.S)


def _strip_tex(text: str) -> str:
    return re.sub(r"\\[a-zA-Z]+|[{}$\\~]", " ", text)


def count(stem: str) -> int:
    """Words in the caption of ``figures/<stem>.pdf`` as the manuscript has it."""
    found = {}
    for path in _DOCUMENTS:
        for match in _FIGURE.finditer(path.read_text()):
            caption = _CAPTION.search(match.group("body"))
            if caption is None:
                raise SystemExit(
                    f"{match.group('stem')} is included in {path.name} with no caption"
                )
            found[match.group("stem")] = len(_strip_tex(caption.group("text")).split())
    if stem not in found:
        # Silence here would let a builder size itself against a caption that no longer
        # exists, which is the failure this module was written to remove.
        raise SystemExit(
            f"{stem} is included by neither the sections nor the "
            f"supplement; found {sorted(found)}"
        )
    return found[stem]


if __name__ == "__main__":
    for path in _DOCUMENTS:
        for match in _FIGURE.finditer(path.read_text()):
            stem = match.group("stem")
            print(f"  {stem:32s} {count(stem):4d} words   ({path.name})")
