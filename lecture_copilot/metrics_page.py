"""Numbers on report pages are generated, never typed: `<span data-metric="a.b.c">…</span>` placeholders are
filled from a results dict (eval/*.json, the decisions table)."""

import html as html_lib
import re

_METRIC = re.compile(r'(<(\w+)\b[^>]*\bdata-metric="([^"]+)"[^>]*>)(.*?)(</\2>)', re.DOTALL)


def _lookup(results: dict, path: str) -> object:
    node: object = results
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{round(v, 4):g}" if abs(v) < 1 else str(round(v, 2))   # 0.0122 stays 0.0122; 94.33 stays
    if v is None:
        return "—"
    return str(v)


def fill_metrics(html: str, results: dict) -> str:
    """Fill every placeholder or raise — a placeholder left unfilled would show a stale number on the page."""
    found = len(re.findall(r"data-metric\s*=", html, re.IGNORECASE))
    parsed = len(_METRIC.findall(html))
    if parsed != found:
        raise ValueError(f"{found - parsed} data-metric placeholder(s) cannot be parsed — "
                         'write them as <tag data-metric="a.b">…</tag> with matching tag case')
    return _METRIC.sub(lambda m: m.group(1) + html_lib.escape(_fmt(_lookup(results, m.group(3))), quote=False)
                       + m.group(5), html)
