"""F-pipeline step 3 deep module: stylometric tell features.

Pure text -> feature dict. Model-independent (MIT), so it lives in the feature
pipeline and is computed identically for human and twin text. No wordlist verdict:
these are distributional stylometrics, because the naive surface tells (em-dash,
"moreover") were shown to NOT separate human from LLM academic prose (BLOCKERS #010).

Scaling/encoding of these features is a model-dependent transform and belongs in
the feature view, not here.
"""

from __future__ import annotations

import re
import statistics

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT = re.compile(r"[^.!?]+[.!?]+", re.S)

# small closed-class function-word set (stylometry leans on these, not topic words)
_FUNCTION = {
    "the", "a", "an", "of", "to", "in", "and", "or", "but", "for", "with", "on",
    "at", "by", "from", "as", "that", "this", "these", "those", "is", "are", "was",
    "were", "be", "been", "it", "its", "we", "our", "they", "their", "which", "such",
    "can", "may", "will", "would", "not", "no", "if", "then", "than", "so", "into",
}
_HEDGE = {"may", "might", "could", "suggest", "suggests", "appear", "appears", "likely",
          "possibly", "perhaps", "seem", "seems", "generally", "often", "typically"}
_BOOSTER = {"clearly", "obviously", "significantly", "substantially", "notably",
            "importantly", "essentially", "particularly", "strongly", "highly"}
_TRANSITION = {"however", "moreover", "furthermore", "additionally", "therefore",
               "thus", "consequently", "hence", "nonetheless", "nevertheless"}


def _rate(n: int, total: int) -> float:
    return round(100.0 * n / total, 4) if total else 0.0


def features(text: str) -> dict:
    """Stylometric feature vector for a section of prose. All values are floats."""
    words = _WORD.findall(text.lower())
    n_words = len(words)
    sents = [s.strip() for s in _SENT.findall(text) if s.strip()]
    sent_lens = [len(_WORD.findall(s)) for s in sents] or [0]

    n_unique = len(set(words))
    long_words = sum(1 for w in words if len(w) > 6)

    return {
        "n_words": float(n_words),
        "n_sentences": float(len(sents)),
        "mean_sent_len": round(statistics.fmean(sent_lens), 4),
        "std_sent_len": round(statistics.pstdev(sent_lens), 4) if len(sent_lens) > 1 else 0.0,
        "ttr": round(n_unique / n_words, 4) if n_words else 0.0,
        "mean_word_len": round(statistics.fmean([len(w) for w in words]), 4) if words else 0.0,
        "pct_long_words": _rate(long_words, n_words),
        "comma_rate": _rate(text.count(","), n_words),
        "semicolon_rate": _rate(text.count(";"), n_words),
        "colon_rate": _rate(text.count(":"), n_words),
        "dash_rate": _rate(text.count("—") + text.count(" - ") + text.count("--"), n_words),
        "paren_rate": _rate(text.count("("), n_words),
        "function_ratio": round(sum(1 for w in words if w in _FUNCTION) / n_words, 4) if n_words else 0.0,
        "hedge_rate": _rate(sum(1 for w in words if w in _HEDGE), n_words),
        "booster_rate": _rate(sum(1 for w in words if w in _BOOSTER), n_words),
        "transition_rate": _rate(sum(1 for w in words if w in _TRANSITION), n_words),
    }


# feature name -> UI description (every feature must be described in the FG)
FEATURE_DOC = {
    "n_words": "Word count of the section",
    "n_sentences": "Sentence count",
    "mean_sent_len": "Mean words per sentence",
    "std_sent_len": "Std dev of sentence length (burstiness; human prose is burstier)",
    "ttr": "Type-token ratio (unique words / total words), lexical diversity",
    "mean_word_len": "Mean word length in characters",
    "pct_long_words": "Percent of words longer than 6 characters",
    "comma_rate": "Commas per 100 words",
    "semicolon_rate": "Semicolons per 100 words",
    "colon_rate": "Colons per 100 words",
    "dash_rate": "Dashes (em/--/spaced hyphen) per 100 words",
    "paren_rate": "Opening parentheses per 100 words",
    "function_ratio": "Fraction of tokens that are closed-class function words",
    "hedge_rate": "Hedge words (may, likely, suggests, ...) per 100 words",
    "booster_rate": "Booster words (clearly, significantly, ...) per 100 words",
    "transition_rate": "Discourse transitions (however, moreover, ...) per 100 words",
}

FEATURE_NAMES = list(FEATURE_DOC.keys())

# categories of tell that live at a specific token (so they can be highlighted in
# the text). The distributional tells (sentence length, lexical diversity, ...)
# have no single locus, so they are deliberately not highlightable.
_HIGHLIGHT_WORDS = [("transition", _TRANSITION), ("booster", _BOOSTER), ("hedge", _HEDGE)]


def highlight_html(text: str, levels: dict | None = None) -> str:
    """HTML-escaped `text` with the locatable lexical tells wrapped in <mark>.
    Only the recognizable, token-level tells (transitions, boosters, hedges, and
    em-dash / semicolon) are marked; distributional tells are not, on purpose.

    `levels` maps a highlight category -> 1|2|3, the strength of that signal in
    this passage (3 = strong marker, 2 = faint wash, 1 = dotted underline for a
    tell that is present but barely moved the score). Absent -> a neutral 2."""
    import html as _html

    def word_class(w: str) -> str | None:
        lw = w.lower()
        for name, vocab in _HIGHLIGHT_WORDS:
            if lw in vocab:
                return name
        return None

    def lvl(cat: str) -> int:
        return (levels or {}).get(cat, 2)

    def mark(cat: str, inner: str) -> str:
        return f"<mark class='hl-{cat} l{lvl(cat)}' title='{cat} tell'>{inner}</mark>"

    out = []
    for m in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z]+", text):
        tok = m.group(0)
        if tok[:1].isalpha():
            cls = word_class(tok)
            esc = _html.escape(tok)
            out.append(mark(cls, esc) if cls else esc)
        else:
            esc = _html.escape(tok)
            esc = esc.replace("—", mark("dash", "—")).replace(";", mark("punc", ";"))
            out.append(esc)
    return "".join(out)


if __name__ == "__main__":
    import json
    import sys

    sample = sys.stdin.read() if not sys.stdin.isatty() else "This is a test. It has two sentences, one comma."
    print(json.dumps(features(sample), indent=2))
