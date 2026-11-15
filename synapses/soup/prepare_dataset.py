#!/usr/bin/env python3
"""Convert Anthropic-Cybersecurity-Skills & grim-prompter markdown into safe ChatML SFT JSONL."""

from __future__ import annotations

import json
import re
from pathlib import Path

SYSTEM_PROMPT = (
    "You are Cynapse Cyber-Rescue, a specialized local cybersecurity and hardware-protection "
    "assistant. Prioritize safe diagnostics, containment, evidence preservation, and non-destructive "
    "remediation. Never execute or recommend destructive commands."
)

DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/(?:\s|$|\*)",
    r"\bmkfs\.[a-z0-9]+",
    r"\bdd\s+if=/dev/(?:zero|urandom)\s+of=/dev/",
    r":\(\)\s*\{\s*:\|\:&\s*\}\s*;:",
    r"\b(?:nc|ncat)\s+.*-e\s+/bin/",
    r"/dev/tcp/[0-9a-zA-Z.-]+/[0-9]+",
    r"\bmsfvenom\b",
    r"\bmimikatz\b",
    r"\bchmod\s+-R\s+777\s+/",
]
DANGEROUS_RE = re.compile("|".join(DANGEROUS_PATTERNS), re.IGNORECASE)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract simple key: value metadata from YAML frontmatter without external deps."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    meta_block = text[4:end]
    body = text[end + 5 :]
    meta: dict[str, str] = {}
    current_key: str | None = None
    for line in meta_block.splitlines():
        if re.match(r"^[a-zA-Z0-9_-]+:\s*", line):
            key, val = line.split(":", 1)
            current_key = key.strip()
            meta[current_key] = val.strip().strip("'\"")
        elif current_key and line.startswith("  ") and not line.strip().startswith("-"):
            meta[current_key] = (meta[current_key] + " " + line.strip()).strip()
    return meta, body


def extract_sections(body: str, wanted: tuple[str, ...], max_chars: int = 1400) -> str:
    """Keep only high-signal defensive sections and cap length for tiny-model context."""
    sections = re.split(r"(?m)^##\s+", body)
    picked: list[str] = []
    for sec in sections[1:]:
        header, _, content = sec.partition("\n")
        h_lower = header.strip().lower()
        if any(w in h_lower for w in wanted):
            cleaned = re.sub(r"\n{3,}", "\n\n", content).strip()
            if cleaned:
                picked.append(f"## {header.strip()}\n{cleaned}")
    joined = "\n\n".join(picked) if picked else body.strip()
    if len(joined) > max_chars:
        joined = joined[:max_chars].rsplit("\n", 1)[0].strip() + "\n..."
    return joined


def is_safe_text(text: str) -> bool:
    return not bool(DANGEROUS_RE.search(text))


def build_record(title: str, prompt: str, response: str, source: str) -> dict | None:
    if not response.strip() or not is_safe_text(prompt) or not is_safe_text(response):
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": response.strip()},
        ],
        "meta": {"title": title, "source": source},
    }


def convert_all(
    anthropic_dir: Path,
    grim_dir: Path,
    output_path: Path,
) -> dict[str, int]:
    records: list[dict] = []
    skipped_unsafe = 0

    if anthropic_dir.exists():
        for skill_md in sorted(anthropic_dir.glob("skills/*/SKILL.md")):
            raw = skill_md.read_text(encoding="utf-8", errors="ignore")
            meta, body = parse_frontmatter(raw)
            name = meta.get("name") or skill_md.parent.name
            desc = meta.get("description") or f"Handle {name.replace('-', ' ')} safely."
            answer = extract_sections(
                body,
                ("overview", "when to use", "workflow", "validation", "remediation", "containment"),
            )
            prompt = (
                f"I suspect an issue related to {name.replace('-', ' ')}. "
                f"How do I safely diagnose, contain, and verify this? Context: {desc}"
            )
            rec = build_record(name, prompt, answer, "anthropic-cybersecurity-skills")
            if rec:
                records.append(rec)
            else:
                skipped_unsafe += 1

    if grim_dir.exists():
        for sub in ("quick", "deep", "methods", "privacy", "skills"):
            sub_dir = grim_dir / sub
            if not sub_dir.exists():
                continue
            for md_file in sorted(sub_dir.rglob("*.md")):
                if md_file.name.lower() == "readme.md":
                    continue
                raw = md_file.read_text(encoding="utf-8", errors="ignore")
                meta, body = parse_frontmatter(raw)
                title = meta.get("title") or meta.get("name") or md_file.stem
                answer = extract_sections(body, ("work", "overview", "checks", "output", "workflow"))
                prompt = f"Run a safe defensive security & threat assessment for: {title}."
                rec = build_record(title, prompt, answer, f"grim-prompter/{sub}")
                if rec:
                    records.append(rec)
                else:
                    skipped_unsafe += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"written": len(records), "skipped_unsafe": skipped_unsafe}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    stats = convert_all(
        Path("/home/orangepi/Documents/reference/skills/Anthropic-Cybersecurity-Skills"),
        Path("/home/orangepi/Documents/reference/skills/grim-prompter"),
        root / "data" / "cyber_rescue_sft.jsonl",
    )
    print(json.dumps(stats, indent=2))
