"""SOTA firewall via the Anthropic SDK (job path).

Same content/prose firewall as firewall.py, but through the Messages API so it
runs as a scalable parallel job instead of the in-pod claude CLI. The writer is
a real within-provider PANEL (opus / sonnet / haiku): different model sizes leave
different fingerprints, so the classifier learns "LLM-ness", not one model's tics.
Sampling params are removed on these models, so panel diversity comes from model
identity, assigned deterministically by pair_id (reproducible, no RNG).
"""

from __future__ import annotations

import hashlib
import json

import anthropic

READER_MODEL = "claude-sonnet-5"  # strong, cheap extraction; completeness is what matters here
WRITER_PANEL = ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"]

SKELETON_SCHEMA = {
    "type": "object",
    "properties": {
        "section_role": {"type": "string"},
        "claims": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"type": "string"}},
        "numbers": {"type": "array", "items": {"type": "string"}},
        "structure": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["section_role", "claims", "entities", "numbers", "structure"],
    "additionalProperties": False,
}

READER_INSTR = """Extract the CONTENT SKELETON of this academic paper section so it can be \
rewritten from scratch WITHOUT the original wording. Capture EVERY factual claim, entity, and \
number so nothing is lost, but copy no phrase or sentence. Facts only, no prose.

INPUT SECTION:
"""

WRITER_INSTR = """Write ONE section of an academic machine-learning paper from this content \
skeleton. You have NEVER seen the original text. Convey exactly the content in the skeleton, in the \
same logical order, as natural fluent academic prose. Output ONLY the section body prose: no \
headings, no lists, no JSON, no commentary.

CONTENT SKELETON:
"""


def pick_writer(pair_id: str) -> str:
    """Deterministic panel assignment by pair_id (reproducible, balanced)."""
    h = int(hashlib.sha256(pair_id.encode()).hexdigest(), 16)
    return WRITER_PANEL[h % len(WRITER_PANEL)]


def read_skeleton(human_text: str, client: anthropic.Anthropic) -> dict:
    """Human prose -> content skeleton (structured JSON, no prose survives)."""
    resp = client.messages.create(
        model=READER_MODEL,
        max_tokens=4096,
        thinking={"type": "disabled"},
        output_config={"format": {"type": "json_schema", "schema": SKELETON_SCHEMA}},
        messages=[{"role": "user", "content": READER_INSTR + human_text}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def write_twin(skeleton: dict, model: str, client: anthropic.Anthropic) -> str:
    """Content skeleton -> LLM twin, authored by `model`. Writer sees ONLY the skeleton."""
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": WRITER_INSTR + json.dumps(skeleton, ensure_ascii=False, indent=2)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()
