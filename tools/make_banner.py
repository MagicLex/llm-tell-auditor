#!/usr/bin/env python3
"""Banner generator for the awesome-ml-systems series.

One SVG banner per repo, same frame, only title/tagline/emoji change. Dark
canvas, emerald accent, the Hopsworks hop-mark as the fixed series brand.
Pure vector (the mark is inlined, not a raster), so it renders on GitHub.

    python tools/make_banner.py \
        --title "README Vaporware Score" \
        --tagline "Predict whether a GitHub repo gets abandoned from its README text alone." \
        --emoji "🚀" --index 001 --out assets/banner.svg
"""
import argparse
from html import escape

W, H = 1280, 360
BG, BORDER = "#0b0e11", "#1c242e"
EMERALD, EMERALD_DIM = "#34d399", "#10b981"
TITLE, TAGLINE, KICKER_DIM = "#f3f4f6", "#9ca3af", "#6b7280"
FONT = "ui-sans-serif,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,'SF Mono','Cascadia Code',Menlo,Consolas,monospace"
PAD = 56

# Hopsworks hop-mark, the 9 green paths lifted from the official 2023 logo SVG.
# Coordinates live in the logo's own space; MARK_VB crops to them.
MARK_VB = "10 13 50 48"
MARK = (
    '<path d="M22.9,58.5c-1.1,0.6-2.3,1-3.5,1.2c-1,0.2-1.9,0.2-2.9,0.2c-1,0-2.1-0.2-3.1-0.3c-0.1,0-0.1,0-0.2-0.1 c0,0,0,0-0.1,0c0,0,0,0,0-0.1c0-0.1-0.1-0.1-0.1-0.2c-0.1-1-0.3-2.1-0.3-3.1c0-1,0-2,0.2-2.9c0.2-1.2,0.6-2.4,1.2-3.5 c0.1-0.2,0.2-0.2,0.3-0.2c1.3,0.2,2.6,0.2,4,0c1.1-0.1,2.2-0.4,3.2-0.9c0.1,0.3,0.1,0.5,0.2,0.8c0,0.4,0.1,0.8,0.4,1 c0.3,0.3,0.7,0.4,1,0.4c0.3,0,0.5,0.1,0.8,0.2c-0.5,1-0.7,2.1-0.9,3.2c-0.2,1.4-0.2,2.7,0,4C23.1,58.4,23,58.5,22.9,58.5z"/>'
    '<path d="M25.1,55c0.1-1.2,0.4-2.4,0.9-3.5c0.1-0.3,0.3-0.4,0.6-0.3c1,0.1,2.1,0,3.1-0.1c1.4-0.2,2.9-0.5,4.3-1 c0.8-0.3,1.5-0.7,2.4-1.1c-0.1,1.6,0.2,3.1,0.7,4.6c0.2,0.5,0.3,0.9,0.6,1.3c0.1,0.2,0.1,0.4-0.1,0.5c-1.6,2-3.5,3.7-5.9,4.7 c-1.7,0.7-3.4,1.2-5.2,1c-0.3,0-0.4-0.1-0.5-0.4C25.3,59,25,57,25.1,55z"/>'
    '<path d="M40.7,55.6c-0.7-1-1.3-2-1.7-3.2c-0.3-1.1-0.5-2.2-0.5-3.5c0.1-2.2,0.7-4.4,1.9-6.5c0.6-1.1,1.4-2.2,2.1-3.2 c0.5-0.8,1.2-1,2-1c0.7,0,1.4,0,2.1-0.2c0.2-0.1,0.4,0,0.6,0.3c0.6,1.6,1.5,3.2,2.4,4.6c0.8,1.3,1.7,2.6,2.7,3.8 c0.1,0.1,0.1,0.2,0,0.4c-0.6,1.2-1.4,2.3-2.2,3.4c-1.2,1.4-2.6,2.5-4.1,3.5c-1.5,0.9-3,1.6-4.7,2.1C41,56.2,40.8,55.8,40.7,55.6z"/>'
    '<path d="M55.3,46.1c1.3-2,2.4-4,3.1-6.3c0.7-2,1.2-5.6,0.7-8.8c-0.5-2.5-1.5-4.8-2.9-7c-1-1.5-2.1-2.9-3.4-4.1 l-0.2-0.2c-1.2-1.2-2.6-2.4-4.1-3.3c-2.1-1.4-4.5-2.4-7-2.9c-3.2-0.5-6.8,0-8.8,0.7c-2.2,0.7-4.3,1.8-6.3,3.1 c-0.3,0.2-0.3,0.3,0,0.5c2.4,2,5,3.8,7.8,5.1c0.5,0.2,1.1,0.5,1.6,0.7c0.8,0.3,1.4,0.6,1.1,1.8c-0.3,1.3-0.5,2.7-0.4,4.1 c0.1,1.7,0.4,3.5,0.9,5.1c0,0.1,0.1,0.2,0.2,0.3c0,0,0,0,0,0.1c0,0,0.1,0,0.1,0c0.1,0.1,0.2,0.1,0.3,0.2c1.7,0.5,3.4,0.8,5.1,0.9 c1.4,0.1,2.7,0,4.1-0.4c1.2-0.3,1.4,0.2,1.8,1.1c0.2,0.5,0.4,1.1,0.7,1.6c1.3,2.9,3.1,5.4,5.1,7.8C55,46.4,55.1,46.4,55.3,46.1z"/>'
    '<path d="M35.8,46.6c-1.6,1.2-3.5,1.8-5.4,2.2c-1.2,0.2-2.4,0.3-3.7,0.3c-0.8,0-1.7-0.1-2.5-0.2c-0.1,0-0.2-0.1-0.3-0.1 c0,0,0,0-0.1,0l0,0c0,0,0,0,0,0c-0.1-0.1-0.1-0.2-0.1-0.3c-0.1-0.8-0.2-1.7-0.2-2.5c0-1.2,0.1-2.5,0.3-3.7c0.3-2,1-3.8,2.2-5.4 c0.4-0.6,0.9-0.9,1.6-1.1c0.9-0.2,1.8-0.6,2.7-1c1.5-0.8,3-1.6,4.3-2.7c0.3,1.4,0.5,2.8,1,4.1c0.1,0.2,0.2,0.4,0.4,0.6l0,0 c0.2,0.2,0.3,0.3,0.6,0.4c1.3,0.5,2.7,0.7,4.1,1c-1.1,1.3-1.9,2.8-2.7,4.3c-0.4,0.9-0.9,1.7-1,2.7C36.8,45.8,36.4,46.2,35.8,46.6z"/>'
    '<path d="M17.6,47.5c1.2-0.1,2.4-0.4,3.5-0.9c0.3-0.1,0.4-0.3,0.3-0.6c-0.1-1,0-2.1,0.1-3.1c0.2-1.4,0.5-2.9,1-4.3 c0.3-0.8,0.7-1.5,1.1-2.4c-1.6,0.1-3.1-0.2-4.6-0.7c-0.5-0.2-0.9-0.3-1.3-0.6c-0.2-0.1-0.4-0.1-0.5,0.1c-2,1.6-3.7,3.5-4.7,5.9 c-0.7,1.7-1.2,3.4-1,5.2c0,0.3,0.1,0.4,0.4,0.5C13.6,47.3,15.6,47.6,17.6,47.5z"/>'
    '<path d="M17,31.9c1,0.7,2,1.3,3.2,1.7c1.1,0.3,2.2,0.5,3.5,0.5c2.2-0.1,4.4-0.7,6.5-1.9c1.1-0.6,2.2-1.4,3.2-2.1 c0.8-0.5,1-1.2,1-2c0-0.7,0-1.4,0.2-2.1c0.1-0.2,0-0.4-0.3-0.6c-1.6-0.6-3.2-1.5-4.6-2.4c-1.3-0.8-2.6-1.7-3.8-2.7 c-0.1-0.1-0.2-0.1-0.4,0c-1.2,0.6-2.3,1.4-3.4,2.2c-1.4,1.2-2.5,2.6-3.5,4.1c-0.9,1.5-1.6,3-2.1,4.7C16.4,31.6,16.8,31.8,17,31.9z"/>'
)
MARK_SIZE = 30


def wrap(text, width=52):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines[:2]


def build(title, tagline, emoji, index, series, link):
    tag_lines = wrap(tagline)
    tspans = "".join(
        f'<tspan x="{PAD}" dy="{0 if i == 0 else 38}">{escape(line)}</tspan>'
        for i, line in enumerate(tag_lines)
    )
    badge = (
        f'<text x="{W - PAD}" y="76" text-anchor="end" font-family="{MONO}" '
        f'font-size="22" fill="{KICKER_DIM}">#{escape(index)}</text>'
        if index else ""
    )
    emoji_tspan = f"{escape(emoji)}&#160;&#160;" if emoji else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">
  <defs>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{EMERALD}"/>
      <stop offset="1" stop-color="{EMERALD_DIM}" stop-opacity="0"/>
    </linearGradient>
    <radialGradient id="glow" cx="0.12" cy="0.1" r="0.9">
      <stop offset="0" stop-color="{EMERALD}" stop-opacity="0.10"/>
      <stop offset="1" stop-color="{EMERALD}" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="{W}" height="{H}" rx="20" fill="{BG}"/>
  <rect width="{W}" height="{H}" rx="20" fill="url(#glow)"/>
  <rect x="1.5" y="1.5" width="{W - 3}" height="{H - 3}" rx="19" fill="none" stroke="{BORDER}" stroke-width="1.5"/>
  <a href="{escape(link)}" target="_blank">
    <svg x="{PAD}" y="50" width="{MARK_SIZE}" height="{MARK_SIZE}" viewBox="{MARK_VB}" fill="{EMERALD}">{MARK}</svg>
    <text x="{PAD + MARK_SIZE + 14}" y="78" font-family="{MONO}" font-size="22" letter-spacing="0.5">
      <tspan fill="{EMERALD}">{escape(series)}</tspan><tspan fill="{KICKER_DIM}"> · hopsworks</tspan>
    </text>
  </a>
  {badge}
  <text x="{PAD}" y="180" font-family="{FONT}" font-size="56" font-weight="700" fill="{TITLE}">{emoji_tspan}{escape(title)}</text>
  <text y="240" font-family="{FONT}" font-size="27" fill="{TAGLINE}">{tspans}</text>
  <rect x="{PAD}" y="{H - 54}" width="180" height="5" rx="2.5" fill="url(#accent)"/>
</svg>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--title", required=True)
    p.add_argument("--tagline", required=True)
    p.add_argument("--emoji", default="")
    p.add_argument("--index", default="", help="series index, e.g. 001")
    p.add_argument("--series", default="awesome-ml-systems")
    p.add_argument("--link", default="https://github.com/MagicLex/awesome-ml-systems")
    p.add_argument("--out", default="assets/banner.svg")
    a = p.parse_args()
    with open(a.out, "w") as f:
        f.write(build(a.title, a.tagline, a.emoji, a.index, a.series, a.link))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
