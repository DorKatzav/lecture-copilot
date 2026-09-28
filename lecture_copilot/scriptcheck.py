"""Foreign-script check for Hebrew/English output (CLAUDE.md, D-M0-10).

The forbidden scripts are listed explicitly — stage 0 saw Cyrillic, Arabic and Vietnamese letters leak into
Hebrew. Accented Latin (é, ü) is allowed: it appears in real names and loanwords.
"""

import re

FORBIDDEN: dict[str, tuple[tuple[int, int], ...]] = {
    "Cyrillic": ((0x0400, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x0870, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "CJK": ((0x1100, 0x11FF), (0x3040, 0x30FF), (0x3130, 0x318F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
            (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0x20000, 0x3134F)),
    "Latin Extended Additional": ((0x1E00, 0x1EFF),),
}


def forbidden_scripts(text: str) -> set[str]:
    found = set()
    for ch in text:
        cp = ord(ch)
        if cp < 0x0400:
            continue
        for name, ranges in FORBIDDEN.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                found.add(name)
    return found


_HEBREW_RUN = re.compile("[\u0590-\u05FF][\u0590-\u05FF\\s\"'.,:;!?-]*")


def terminal_text(text: str, limit: int = 160) -> str:
    """One line for the terminal, Hebrew replaced by [he] (CLAUDE.md: Hebrew never goes to the terminal).
    Provider messages can be Hebrew — macOS localizes MacWhisper's errors to the system language."""
    return " ".join(_HEBREW_RUN.sub("[he] ", text).split())[:limit]
