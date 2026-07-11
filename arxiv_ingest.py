"""arXiv ingestion for the LLM Tell Auditor (#010).

Fetch preprint LaTeX e-print source, parse to clean prose per section.
The deep module here is `parse_sections`: LaTeX -> [(section_title, prose)].
It is pure and testable in isolation; the network functions wrap it.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from pylatexenc.latex2text import LatexNodes2Text

ARXIV_API = "https://export.arxiv.org/api/query"
EPRINT_URL = "https://arxiv.org/e-print/"
_UA = {"User-Agent": "hopsworks-tell-auditor/0.1 (research; contact adminaccounts@hopsworks.ai)"}

_CONVERTER = LatexNodes2Text(math_mode="remove", keep_comments=False, strict_latex_spaces=False)

# environments whose content is not prose: drop wholesale before text conversion
_ENV_DROP = re.compile(
    r"\\begin\{(figure|figure\*|table|table\*|equation|equation\*|align|align\*|"
    r"tabular|algorithm|algorithmic|thebibliography|lstlisting|verbatim)\}"
    r".*?\\end\{\1\}",
    re.S,
)
_SECTION = re.compile(r"\\section\*?\s*\{([^}]*)\}")
_ABSTRACT = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.S)
_COMMENT = re.compile(r"(?<!\\)%.*")
_DOCBODY = re.compile(r"\\begin\{document\}(.*?)\\end\{document\}", re.S)
_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")


def _strip_comments(tex: str) -> str:
    return _COMMENT.sub("", tex)


# pylatexenc renders \cite/\ref/\url/subsection markers as these literals: noise for prose
_MARKERS = re.compile(r"<cit\.>|<ref>|<>|§+(?:\.§+)*")


def _to_text(latex: str) -> str:
    latex = _ENV_DROP.sub(" ", latex)
    try:
        text = _CONVERTER.latex_to_text(latex)
    except Exception:
        text = latex
    text = _MARKERS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_sections(tex: str) -> list[tuple[str, str]]:
    """LaTeX body -> list of (section_title, clean_prose). Deep module, pure."""
    tex = _strip_comments(tex)
    body_match = _DOCBODY.search(tex)
    body = body_match.group(1) if body_match else tex

    sections: list[tuple[str, str]] = []

    abs_match = _ABSTRACT.search(body)
    if abs_match:
        abstract = _to_text(abs_match.group(1))
        if abstract:
            sections.append(("Abstract", abstract))

    matches = list(_SECTION.finditer(body))
    for i, m in enumerate(matches):
        title = _to_text(m.group(1)).strip() or f"Section {i + 1}"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        prose = _to_text(body[start:end])
        if len(prose.split()) >= 20:  # skip stubs / acknowledgements
            sections.append((title, prose))
    return sections


def _inline_inputs(main: str, files: dict[str, str]) -> str:
    """Best-effort one-level inline of \\input/\\include includes."""

    def repl(m: re.Match) -> str:
        name = m.group(1).strip()
        for key in (name, name + ".tex", name.replace("./", "")):
            if key in files:
                return files[key]
        return ""

    return _INPUT.sub(repl, main)


def extract_main_tex(source_bytes: bytes) -> str | None:
    """Extract the main .tex (the one with \\begin{document}) from an e-print blob.

    The blob is either a gzipped tar of the project, or a single gzipped .tex.
    """
    tex_files: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(source_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".tex"):
                    f = tar.extractfile(member)
                    if f is not None:
                        tex_files[member.name] = f.read().decode("utf-8", "ignore")
    except tarfile.ReadError:
        try:
            single = gzip.decompress(source_bytes).decode("utf-8", "ignore")
        except OSError:
            return None
        tex_files["main.tex"] = single

    if not tex_files:
        return None

    main_name = next(
        (n for n, t in tex_files.items() if "\\begin{document}" in t),
        max(tex_files, key=lambda n: len(tex_files[n])),
    )
    return _inline_inputs(tex_files[main_name], tex_files)


def fetch_source(arxiv_id: str, retries: int = 3) -> bytes | None:
    url = EPRINT_URL + arxiv_id
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(3 * (attempt + 1))
    return None


def fetch_recent_ids(category: str, max_results: int = 50) -> list[dict]:
    """Return recent papers in a category: [{id, title, category, published}]."""
    q = urllib.parse.urlencode(
        {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    req = urllib.request.Request(f"{ARXIV_API}?{q}", headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        xml = r.read()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in ET.fromstring(xml).findall("a:entry", ns):
        raw_id = entry.findtext("a:id", default="", namespaces=ns)
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        out.append(
            {
                "paper_id": arxiv_id,
                "title": (entry.findtext("a:title", "", ns) or "").strip(),
                "category": category,
                "published": entry.findtext("a:published", "", ns),
            }
        )
    return out


if __name__ == "__main__":
    import sys

    ident = sys.argv[1] if len(sys.argv) > 1 else "2607.08754"
    blob = fetch_source(ident)
    if not blob:
        print(f"no source for {ident}")
        sys.exit(1)
    main_tex = extract_main_tex(blob)
    if not main_tex:
        print(f"no main tex for {ident}")
        sys.exit(1)
    for title, prose in parse_sections(main_tex):
        print(f"\n### {title}  ({len(prose.split())} words)")
        print(prose[:280])
