"""Generate RAD `analysis` and `guidance` fields for a database CSV."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .interfaces import LanguageModel

ANNOTATION_SYSTEM_PROMPT = """You are a Strategy-and-Guidance Generator in a retrieval-augmented defense pipeline.
Given a jailbreak prompt and its underlying intent, your role is to describe the general pattern of
disguise used, and provide reusable guidance that can help extractors uncover the true intent from
any similar prompt.

Core rules:
1. Generalization: Strategies must describe a reusable disguise pattern, not case-specific details.
2. Guidance must provide broadly applicable heuristics for recovering intent from prompts of that pattern.
3. Strategy: Write one concise sentence naming and describing the disguise type.
4. Guidance: Write one to three short bullet heuristics that could apply to any prompt using this pattern.
5. Cues: List one to three abstract cues that often signal this pattern, not just literal strings.
6. Applicability: State whether the strategy clearly applies (Applicable | Not applicable | Unclear).
7. Confidence: State confidence (High | Medium | Low).
8. Safety: Never restate operationally harmful instructions. Keep outputs abstract.
9. Output format: Use the tagged format exactly, with no extra commentary."""


ANNOTATION_PROMPT = """Input:
User Prompt: {prompt}
User Intent: {intent}

Your Tasks:
1. Compare the prompt and the intent, and identify the disguise pattern used to turn the intent into the prompt.
2. Write a one-sentence Strategy describing this pattern in general terms (not tied to this single prompt).
3. Provide Guidance: 1–3 short bullet heuristics that would help an extractor uncover the true intent from any prompt using this pattern.
4. List 1–3 abstract Cues that typically signal this pattern. Do not copy unique phrases directly from the input prompt.
5. Mark Applicability (Applicable | Not applicable | Unclear).
6. State your Confidence level (High | Medium | Low).

Output Format (use these exact section headers):

Strategy: <one general sentence>

Guidance:
- <heuristic 1>
- <heuristic 2> (optional)
- <heuristic 3> (optional)

Cues: <cue1>; <cue2>; <cue3>

Applicability: <Applicable | Not applicable | Unclear>

Confidence: <High | Medium | Low>"""


@dataclass(frozen=True)
class Annotation:
    analysis: str
    guidance: str
    cues: str
    applicability: str
    confidence: str


def _section(text: str, heading: str, following: tuple[str, ...]) -> str:
    marker = f"{heading}:"
    following_markers = tuple(f"{item}:" for item in following)
    lines = text.splitlines()
    for position, line in enumerate(lines):
        if line.strip().casefold().startswith(marker.casefold()):
            value = line.split(":", 1)[1].strip()
            remainder = [value] if value else []
            for following_line in lines[position + 1 :]:
                if any(
                    following_line.strip().casefold().startswith(item.casefold())
                    for item in following_markers
                ):
                    break
                remainder.append(following_line)
            value = "\n".join(remainder).strip()
            if not value:
                raise ValueError(f"annotation has an empty {heading}:")
            return value
    raise ValueError(f"annotation is missing {heading}:")


def parse_annotation(text: str) -> Annotation:
    """Parse and completely validate the documented tagged model response."""

    headings = ("Strategy", "Guidance", "Cues", "Applicability", "Confidence")
    lines = text.splitlines()
    positions: list[int] = []
    for heading in headings:
        matches = [
            index
            for index, line in enumerate(lines)
            if line.strip().casefold().startswith(f"{heading}:".casefold())
        ]
        if len(matches) != 1:
            raise ValueError(f"annotation must contain exactly one {heading}: section")
        positions.append(matches[0])
    if positions != sorted(positions):
        raise ValueError("annotation sections are out of order")
    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty or not nonempty[0].casefold().startswith("strategy:"):
        raise ValueError("annotation contains commentary before Strategy:")

    sections = {
        heading: _section(text, heading, headings[index + 1 :])
        for index, heading in enumerate(headings)
    }
    if "\n" in sections["Strategy"]:
        raise ValueError("annotation Strategy must be one sentence on one line")
    guidance_lines = [
        line.strip() for line in sections["Guidance"].splitlines() if line.strip()
    ]
    if not 1 <= len(guidance_lines) <= 3 or any(
        not line.startswith("- ") or not line[2:].strip() for line in guidance_lines
    ):
        raise ValueError(
            "annotation Guidance must contain one to three non-empty bullets"
        )
    cues = [cue.strip() for cue in sections["Cues"].split(";")]
    if not 1 <= len(cues) <= 3 or any(not cue for cue in cues):
        raise ValueError(
            "annotation Cues must contain one to three semicolon-separated cues"
        )
    applicability = sections["Applicability"].strip().casefold()
    allowed_applicability = {
        "applicable": "Applicable",
        "not applicable": "Not applicable",
        "unclear": "Unclear",
    }
    if applicability not in allowed_applicability:
        raise ValueError(
            "annotation Applicability must be Applicable, Not applicable, or Unclear"
        )
    confidence = sections["Confidence"].strip().casefold()
    allowed_confidence = {"high": "High", "medium": "Medium", "low": "Low"}
    if confidence not in allowed_confidence:
        raise ValueError("annotation Confidence must be High, Medium, or Low")
    return Annotation(
        analysis=sections["Strategy"],
        guidance=sections["Guidance"],
        cues=sections["Cues"],
        applicability=allowed_applicability[applicability],
        confidence=allowed_confidence[confidence],
    )


def annotate_rows(
    rows: Iterable[dict[str, str]], model: LanguageModel
) -> list[dict[str, str]]:
    """Add validated annotation fields to database-like rows using a local language model."""

    completed: list[dict[str, str]] = []
    for number, row in enumerate(rows, start=2):
        query = (row.get("query") or "").strip()
        jailbreak = (row.get("jailbreak") or "").strip()
        if not query or not jailbreak:
            raise ValueError(
                f"input row {number} requires non-empty query and jailbreak"
            )
        generated = model.generate(
            [ANNOTATION_PROMPT.format(prompt=jailbreak, intent=query)],
            system_prompt=ANNOTATION_SYSTEM_PROMPT,
        )
        if len(generated) != 1:
            raise ValueError(
                "annotation model returned an unexpected number of outputs"
            )
        annotation = parse_annotation(generated[0])
        completed.append(
            {
                **row,
                "query": query,
                "jailbreak": jailbreak,
                "analysis": annotation.analysis,
                "guidance": annotation.guidance,
            }
        )
    if not completed:
        raise ValueError("input CSV contains no rows")
    return completed


def annotate_csv(path: str | Path, output: str | Path, model: LanguageModel) -> Path:
    """Read `query,jailbreak` and write a RAD database CSV with generated annotations."""

    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"query", "jailbreak"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{source} is missing columns: {', '.join(sorted(missing))}"
            )
        rows = annotate_rows(reader, model)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "example_id",
        "query",
        "jailbreak",
        "analysis",
        "guidance",
        "attack_family",
        "split",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    fields.extend(field for field in rows[0] if field not in fields)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination
