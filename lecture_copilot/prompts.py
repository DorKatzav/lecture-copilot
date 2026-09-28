"""Versioned prompt files: `prompts/<name>.md` with `## system` and `## user` sections and `{{var}}` placeholders.

Rendering is strict both ways — a missing or an unknown variable raises — so a renamed placeholder fails loudly.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from lecture_copilot.config import PROMPTS_DIR

_SECTION = re.compile(r"^## (system|user)\s*$", re.MULTILINE)
_VAR = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class Prompt:
    name: str
    system: str
    user: str

    @property
    def variables(self) -> set[str]:
        return set(_VAR.findall(self.system)) | set(_VAR.findall(self.user))

    def render(self, **values: object) -> tuple[str, str]:
        missing = self.variables - values.keys()
        unknown = values.keys() - self.variables
        if missing or unknown:
            raise KeyError(f"{self.name}: missing={sorted(missing)} unknown={sorted(unknown)}")

        def sub(text: str) -> str:
            return _VAR.sub(lambda m: str(values[m.group(1)]), text)

        return sub(self.system), sub(self.user)


def load(name: str, directory: Path = PROMPTS_DIR) -> Prompt:
    text = (directory / f"{name}.md").read_text(encoding="utf-8")
    parts = _SECTION.split(text)  # [title, "system", body, "user", body]
    sections = dict(zip(parts[1::2], parts[2::2], strict=True))
    return Prompt(name=name, system=sections["system"].strip(), user=sections["user"].strip())
