"""On-the-fly plain-language feedback for a scored piece of text (Anthropic).

The classifier returns a probability and 16 stylometric numbers. Most people do
not know what "type-token ratio" or "function-word fraction" mean, so this turns
the fired tells into a short, honest, grounded explanation.

The hard constraint is the whole project's ethic: SIGNAL, NOT VERDICT. The model
is told, in the strongest terms, never to declare the text AI-written or human-
written, never to judge quality, to ground every claim in the numbers it is
given, and to flag when the input is outside the academic-prose domain the
classifier was trained on (where the score means little).
"""

from __future__ import annotations

import json

import anthropic

EXPLAIN_MODEL = "claude-sonnet-5"  # capable enough to hold the honest framing, cheap on intro pricing

SYSTEM = """You explain a stylometric writing-tell auditor to a curious non-expert.

The auditor scores academic prose for how much its STYLE resembles LLM-authored \
rewrites of arXiv paper sections. It looks only at distributional style (sentence \
length, word length, punctuation rates, lexical diversity), never at meaning.

Absolute rules, never break them:
- SIGNAL, NOT VERDICT. Never say or imply the text "is AI-generated", "was written \
by a human", or "used AI". Say only that its style "matches" or "does not match" \
known LLM writing patterns. A high score is a stylistic resemblance, not authorship.
- Never judge quality. Do not call the writing good, bad, strong, weak, or the work \
good or bad science. You measure a style dial, nothing else.
- Ground every sentence in the numbers you are given. Name the tell, its measured \
value, and what that value tends to indicate stylistically. No hand-waving.
- If the input is clearly NOT academic prose (a tweet, an email, code, a few words), \
say so plainly: the model was trained on arXiv sections, so its score is out of \
domain and should not be trusted here.
- Naive detectors over-flag non-native English writers. If the score is high, note \
this so the reader does not take it as an accusation.

Write 110-170 words, plain language, no markdown headings, no bullet lists, no jargon \
left unexplained. Address the reader as "you" when it is their own text."""

# feature -> one plain phrase for what a HIGH value tends to mean (grounds the model)
_DIRECTION = {
    "mean_sent_len": "longer sentences",
    "mean_word_len": "longer words",
    "pct_long_words": "more long (6+ char) words, denser jargon",
    "std_sent_len": "more varied sentence lengths (human prose is usually burstier, so LOW here leans LLM)",
    "ttr": "higher lexical variety (LOW ttr, more repetition, leans LLM)",
    "n_sentences": "more sentences",
    "n_words": "a longer passage",
    "comma_rate": "more commas",
    "semicolon_rate": "more semicolons",
    "colon_rate": "more colons",
    "dash_rate": "more dashes",
    "paren_rate": "more parenthetical asides (fewer of these leans LLM)",
    "function_ratio": "more function words",
    "hedge_rate": "more hedging (may, likely, suggests)",
    "booster_rate": "more boosters (clearly, significantly)",
    "transition_rate": "more discourse transitions (however, moreover)",
}


def _brief(result: dict) -> str:
    """Compact factual brief of the score and the tells that moved it."""
    tells = result.get("all_tells") or result.get("top_tells") or []
    toward_llm = [t for t in tells if t["contribution"] > 0][:6]
    toward_human = sorted((t for t in tells if t["contribution"] < 0),
                          key=lambda t: t["contribution"])[:3]

    def fmt(t):
        d = _DIRECTION.get(t["tell"], t.get("doc", ""))
        return f"- {t['tell']} = {t['value']} (pushes {'LLM' if t['contribution']>0 else 'human'} {t['contribution']:+.2f}); high means {d}"

    lines = [
        f"P(matches LLM style) = {result['proba']:.2f} (>=0.50 is 'flagged').",
        f"Passage length: {result.get('n_words','?')} words.",
        "Tells pushing toward LLM style:",
        *[fmt(t) for t in toward_llm],
        "Tells pushing toward human style:",
        *[fmt(t) for t in toward_human],
    ]
    return "\n".join(lines)


def _user_prompt(text: str, result: dict) -> str:
    excerpt = text.strip()[:800]
    return (f"{_brief(result)}\n\nThe passage (start):\n\"\"\"\n{excerpt}\n\"\"\"\n\n"
            "Explain to the reader, in plain language, what these signals mean for "
            "this passage. Follow every rule.")


def explain(text: str, result: dict, client: anthropic.Anthropic,
            model: str = EXPLAIN_MODEL) -> str:
    """Return a short plain-language explanation of the scored result. Raises on
    API error; callers should degrade gracefully (show the score without it)."""
    resp = client.messages.create(
        model=model,
        max_tokens=700,
        thinking={"type": "disabled"},
        system=SYSTEM,
        messages=[{"role": "user", "content": _user_prompt(text, result)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def explain_stream(text: str, result: dict, client: anthropic.Anthropic,
                   model: str = EXPLAIN_MODEL):
    """Yield the explanation as text deltas, for a streamed UI. Raises on API
    error; callers should catch and degrade."""
    with client.messages.stream(
        model=model,
        max_tokens=700,
        thinking={"type": "disabled"},
        system=SYSTEM,
        messages=[{"role": "user", "content": _user_prompt(text, result)}],
    ) as stream:
        for delta in stream.text_stream:
            yield delta


if __name__ == "__main__":
    import os
    import sys

    import auditor
    mdir = os.environ.get("MODEL_DIR")
    if not mdir:
        import hopsworks
        mdir = hopsworks.login().get_model_registry().get_model("tell_classifier", version=auditor.MODEL_VERSION).download()
    aud = auditor.load_auditor(mdir)
    txt = sys.stdin.read() if not sys.stdin.isatty() else (
        "We propose a novel framework that leverages a comprehensive suite of "
        "techniques to significantly improve performance across diverse benchmarks. "
        "Moreover, our extensive experiments demonstrate substantial gains.")
    res = auditor.audit_text(txt, aud)
    client = anthropic.Anthropic()
    print(f"\nP(LLM style)={res['proba']}  words={res['n_words']}\n")
    print(explain(txt, res, client))
