"""LLM Tell Auditor -- server-rendered viewer + streaming live auditor (no SPA).

Base experience is fully server-rendered and JS-free: browse precomputed
`paper_dossiers`, and a plain `POST /audit` returns a complete result page
(crawlable, works with JS off). On top of that, a progressive enhancement: if JS
is on, the form streams instead, scoring item by item (sections for a paper,
paragraphs for pasted text) with rows that slide into a document pane, stabilo
highlights on the flagged words, and the plain-language feedback typed in as the
LLM writes it.

One review layout everywhere: document on the left, a sticky score rail on the
right. Stored dossiers, live audits, and the no-JS fallback all render through
the same builders, so the three paths cannot drift apart.

Signal, not verdict, enforced in the explanation prompt. One tell family
(stylometric polish). Only token-level tells are highlighted; distributional
ones have no single locus and are left unmarked on purpose.
"""
import asyncio
import html
import json
import math
import os
import re
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import anthropic  # noqa: E402
import auditor as A  # noqa: E402
import explain as E  # noqa: E402
from tell_features import highlight_html  # noqa: E402

MAX_CHARS = 8000

DOSSIERS = {"by_id": {}, "rows": [], "at": 0.0, "error": ""}
ENGINE = {"auditor": None, "client": None, "ready": False, "note": ""}
REFRESH_S = 600

CSS = """
*{box-sizing:border-box} body{margin:0;background:#F4F5F7;color:#1B2430;
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
a{color:#0B7C5D;text-decoration:none} a:hover{text-decoration:underline}
.wrap{max-width:1060px;margin:0 auto;padding:24px 18px 64px}
h1{font-size:1.5rem;margin:0 0 2px} h2{font-size:1.05rem;margin:24px 0 10px}
.sub{color:#5D6875;margin:0 0 16px}
.band{border-left:5px solid #0E9A73;background:#EDF4F0;padding:12px 16px;
 border-radius:6px;margin:0 0 20px;font-size:.92rem;color:#41505E} .band b{color:#1B2430}
form.audit{margin:0 0 10px} textarea{width:100%;min-height:120px;background:#FFFFFF;
 color:#1B2430;border:1px solid #D5DBE2;border-radius:8px;padding:12px;font:inherit;resize:vertical}
.row{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
button{background:#0E9A73;color:#fff;border:none;font-weight:700;padding:9px 18px;
 border-radius:8px;cursor:pointer;font:inherit} button:hover{background:#0B8563}
button:disabled{opacity:.5;cursor:default}
.hint{color:#5D6875;font-size:.85rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:0 0 18px}
.tile{background:#FFFFFF;border:1px solid #E3E7EC;border-radius:10px;padding:12px 14px}
.tile .v{font-size:1.7rem;font-weight:700;line-height:1.2} .tile .l{color:#5D6875;font-size:.8rem}
.cats{margin:0 0 12px} .cats a{display:inline-block;background:#FFFFFF;border:1px solid #D5DBE2;
 color:#41505E;padding:2px 10px;margin:2px 6px 2px 0;border-radius:12px;font-size:.8rem}
.cats a.on{border-color:#0E9A73;color:#0B7C5D}
table{width:100%;border-collapse:collapse;font-size:.92rem}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #E3E7EC}
th{color:#5D6875;font-weight:600} th a{color:inherit} th a.on{color:#1B2430} tr:hover td{background:#EDF0F3}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{display:inline-block;width:90px;height:8px;border-radius:4px;background:#E3E7EC;
 vertical-align:middle;overflow:hidden}
.bar i{display:block;height:100%;border-radius:4px;background:#2E8F70;transition:width .5s ease}
.bar.hot i{background:#C0394B}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.78rem;font-weight:600}
.flag{background:#FBE9EB;color:#A32536} .ok{background:#E6F5EE;color:#14804A}
.warn{background:#FBF3DC;color:#8A6D1A}
.pill{display:inline-block;background:#F0F2F5;border:1px solid #DDE2E8;color:#41505E;
 padding:1px 8px;margin:2px 4px 2px 0;border-radius:10px;font-size:.8rem}
.pill[data-spot]{cursor:pointer} .pill[data-spot]:hover{border-color:#AAB4BF}
.pill.on{border-color:#0E9A73;color:#0B7C5D}
.contrib{color:#A32536} .neg{color:#2A6FA8} .muted{color:#5D6875}
.review{background:#EDF6F1;border:1px solid #CBE3D7;border-radius:10px;padding:14px 16px;margin:6px 0;
 white-space:pre-wrap;line-height:1.6;font-size:.92rem}
.score{font-size:2.2rem;font-weight:800}
.foot{color:#7A8592;font-size:.82rem;margin-top:36px;border-top:1px solid #E3E7EC;padding-top:14px}
/* stabilo highlights, layered by signal strength: l3 marker, l2 wash, l1 underline */
mark[class^=hl-]{padding:0 .12em;border-radius:3px;box-decoration-break:clone;
 -webkit-box-decoration-break:clone;background:none;color:inherit}
.l3{color:#141A20;font-weight:600}
.hl-transition.l3{background:#ffe14d} .hl-transition.l2{background:rgba(255,225,77,.45)} .hl-transition.l1{border-bottom:2px dotted #d9b902}
.hl-booster.l3{background:#ff9de0} .hl-booster.l2{background:rgba(255,157,224,.40)} .hl-booster.l1{border-bottom:2px dotted #e35db8}
.hl-hedge.l3{background:#7df0b2} .hl-hedge.l2{background:rgba(125,240,178,.40)} .hl-hedge.l1{border-bottom:2px dotted #1eae66}
.hl-dash.l3{background:#8fd0ff} .hl-dash.l2{background:rgba(143,208,255,.45)} .hl-dash.l1{border-bottom:2px dotted #2f8ed6}
.hl-punc.l3{background:#ffc07a} .hl-punc.l2{background:rgba(255,192,122,.45)} .hl-punc.l1{border-bottom:2px dotted #e08b2d}
.legend{font-size:.8rem;color:#5D6875;margin:6px 0 2px} .legend mark{margin-right:2px}
/* spotlight: clicking a locatable tell pill dims every mark except that family */
.doc[data-spot] mark[class^=hl-]{background:none;border-bottom:none;color:inherit;font-weight:400;opacity:.85}
.doc[data-spot=transition] mark.hl-transition{background:#ffe14d;color:#141A20;font-weight:600;opacity:1}
.doc[data-spot=booster] mark.hl-booster{background:#ff9de0;color:#141A20;font-weight:600;opacity:1}
.doc[data-spot=hedge] mark.hl-hedge{background:#7df0b2;color:#141A20;font-weight:600;opacity:1}
.doc[data-spot=dash] mark.hl-dash{background:#8fd0ff;color:#141A20;font-weight:600;opacity:1}
.doc[data-spot=punc] mark.hl-punc{background:#ffc07a;color:#141A20;font-weight:600;opacity:1}
/* streaming polish */
#progress{color:#5D6875;font-size:.85rem;margin:8px 0;min-height:1.2em}
.cursor::after{content:"\\258c";color:#0E9A73;animation:blink 1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
/* two-pane review: document left, sticky score+review rail right, scan sweep */
.stage{display:grid;grid-template-columns:1fr 320px;gap:18px;align-items:start;margin-top:10px}
.doc{position:relative;overflow:hidden;border:1px solid #E3E7EC;border-radius:10px;background:#FFFFFF;
 box-shadow:0 1px 3px rgba(27,36,48,.06)}
.doc.scanning::after{content:"";position:absolute;left:0;right:0;top:-100px;height:100px;pointer-events:none;
 background:linear-gradient(180deg,transparent,rgba(14,154,115,.14),transparent);animation:scan 1.9s linear infinite}
@keyframes scan{from{top:-100px}to{top:100%}}
.drow{padding:13px 15px;border-bottom:1px solid #EEF1F4;border-left:3px solid transparent}
.drow:last-child{border-bottom:none} .drow.flagged{border-left-color:#C0394B}
.dtitle{font-weight:600;font-size:.85rem;color:#5D6875;margin:0 0 5px}
.dtext{line-height:1.85;font-size:1.02rem;color:#232D3A;font-family:Charter,Georgia,'Times New Roman',serif}
.dnote{margin-top:7px;font-size:.82rem;color:#68737F;display:flex;gap:8px;align-items:baseline}
.dnote.flagged{color:#A32536} .dnote .p{font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}
.rail{position:sticky;top:14px;align-self:start} #ov{margin:0 0 8px}
.dial{position:relative;width:110px;height:110px;margin:0 0 6px}
.dial svg{transform:rotate(-90deg)}
.dial .n{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
 font-size:1.55rem;font-weight:800}
.rail h3{margin:14px 0 6px;font-size:.95rem}
.toc{margin:10px 0 0;font-size:.82rem}
.toc a{display:flex;justify-content:space-between;gap:8px;color:#41505E;padding:3px 6px;
 border-radius:6px;border-left:2px solid transparent}
.toc a:hover{background:#EDF0F3;text-decoration:none} .toc a.f{border-left-color:#C0394B;color:#A32536}
.toc .p{font-variant-numeric:tabular-nums;white-space:nowrap}
@media(max-width:760px){.stage{grid-template-columns:1fr}.rail{position:static}}
"""

FAVICON = ("<link rel=icon href=\"data:image/svg+xml,"
           "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
           "<rect width='100' height='100' rx='18' fill='%23ffe14d'/>"
           "<text x='50' y='72' font-size='62' text-anchor='middle'>&#128269;</text></svg>\">")

LEGEND = ("<div class=legend>Highlights: <mark class='hl-transition l3'>however</mark>"
          " transition <mark class='hl-booster l3'>clearly</mark> booster "
          "<mark class='hl-hedge l3'>may</mark> hedge <mark class='hl-dash l3'>&mdash;</mark> em-dash "
          "<mark class='hl-punc l3'>;</mark> semicolon &nbsp;&middot;&nbsp; "
          "brighter marker = stronger signal, dotted underline = present but weak</div>")

BANNER = ('<div class="band"><b>Signal, not verdict.</b> This reports which known '
          '<b>LLM writing tells</b> a passage matches, with the measured evidence. It is '
          '<b>not an AI detector</b> and not a quality judgement. One family is scored, '
          'stylometric polish. Non-native English is over-flagged by naive detectors, so '
          'we publish evidence, not accusations.</div>')

FORM = ('<form class="audit" method="post" action="{base}/audit" id="af">'
        '<textarea name="q" id="q" placeholder="Paste an arXiv id or URL (e.g. 2607.08754), '
        'or paste any prose to audit its style..."></textarea>'
        '<div class="row"><button type="submit" id="go">Audit</button>'
        '<span class="hint">Scored live, item by item, with a plain-language explanation.</span>'
        '</div></form>')


def _base(req: Request) -> str:
    return (req.scope.get("root_path") or "").rstrip("/")


def _e(x) -> str:
    return html.escape(str(x))


def _bar(p: float, hot: bool | None = None) -> str:
    hot = p >= A.FLAG_THRESHOLD if hot is None else hot
    return (f"<span class='bar{' hot' if hot else ''}'>"
            f"<i style='width:{max(2, round(p * 100))}%'></i></span>")


_DIAL_C = 2 * math.pi * 50


def _dial(p: float, hot: bool) -> str:
    col = "#C0394B" if hot else "#0E9A73"
    return (f"<div class=dial><svg width=110 height=110 viewBox='0 0 110 110'>"
            f"<circle cx=55 cy=55 r=50 fill=none stroke='#E3E7EC' stroke-width=10 />"
            f"<circle cx=55 cy=55 r=50 fill=none stroke='{col}' stroke-width=10 "
            f"stroke-linecap=round stroke-dasharray={_DIAL_C:.1f} "
            f"stroke-dashoffset={_DIAL_C * (1 - min(1.0, p)):.1f} /></svg>"
            f"<div class=n>{p:.2f}</div></div>")


def _page(title: str, body: str, script: str = "", desc: str = "") -> str:
    og = (f"<meta property=og:title content='{_e(title)}'>"
          f"<meta property=og:description content='{_e(desc)}'>") if desc else ""
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title>{FAVICON}{og}<style>{CSS}</style></head>"
            f"<body><div class=wrap>{body}</div>{SPOT_SCRIPT}{script}</body></html>")


def _pill(tell: str, tail: str, doc: str = "") -> str:
    """One tell pill. Locatable tells are clickable (spotlight in the document);
    every pill gets its plain-language description as a tooltip."""
    cat = A._HL_FEATURE.get(tell)
    tip = doc + (". Click to spotlight it in the text." if cat and doc else "")
    attrs = (f" title='{_e(tip)}'" if tip else "") + (f" data-spot={cat}" if cat else "")
    return f"<span class=pill{attrs}>{_e(tell)} {tail}</span>"


def _tell_pills(tells) -> str:
    return "".join(
        _pill(t["tell"],
              f"<span class={'contrib' if t['contribution'] > 0 else 'neg'}>{t['contribution']:+.2f}</span> "
              f"<span class=muted>({t['value']})</span>", t.get("doc", ""))
        for t in tells)


def _drive_pills(tells) -> str:
    return "".join(_pill(t["tell"], f"<b class=contrib>+{t['drive']:.2f}</b>", t.get("doc", ""))
                   for t in tells)


# plain phrase per tell for a margin comment (only the human-legible ones)
_PHRASE = {
    "mean_sent_len": "long sentences", "mean_word_len": "long words",
    "pct_long_words": "dense, jargon-heavy words", "std_sent_len": "very even sentence lengths",
    "ttr": "little word variety", "paren_rate": "few asides", "function_ratio": "formal phrasing",
    "transition_rate": "discourse transitions", "booster_rate": "booster words",
    "hedge_rate": "hedging", "comma_rate": "comma use", "semicolon_rate": "semicolons",
    "colon_rate": "colons", "dash_rate": "dashes", "n_sentences": "length", "n_words": "length",
}


def _comment(it: dict) -> str:
    """A short, honest margin note for a passage: never a verdict. top_tells hold
    the LLM-leaning signals present; when the passage still reads human we frame
    them as outweighed rather than pretending they drove it."""
    tells: list[str] = []
    for t in it.get("top_tells", [])[:3]:
        ph = _PHRASE.get(t["tell"])
        if ph and ph not in tells:
            tells.append(ph)
    why = ", ".join(tells)
    if it["flagged"]:
        return f"Matches LLM style ({it['proba']:.2f}). {why[0].upper() + why[1:]}." if why \
            else f"Matches LLM style ({it['proba']:.2f})."
    if why:
        return f"Reads human ({it['proba']:.2f}). Some LLM-ish notes ({why}), but outweighed."
    return f"Reads human ({it['proba']:.2f})."


# --- the one review layout: document pane + score rail --------------------
def _doc_row(i: int, it: dict, body_html: str) -> str:
    flagged = it["flagged"]
    title = it.get("title") or ""
    dtitle = f"<div class=dtitle>{_e(title)}</div>" if title and not title.startswith("Paragraph") else ""
    return (f"<div class='drow{' flagged' if flagged else ''}' id=s{i}>{dtitle}"
            f"<div class=dtext>{body_html}</div>"
            f"<div class='dnote{' flagged' if flagged else ''}'>"
            f"<span class=p>{_bar(it['proba'], flagged)} {it['proba']:.2f}</span>"
            f"<span>{it.get('comment') or _comment(it)}</span></div></div>")


def _toc(sections) -> str:
    links = "".join(
        f"<a href='#s{i}' class='{'f' if s['flagged'] else ''}'>"
        f"<span>{_e((s.get('title') or f'Passage {i + 1}')[:34])}</span>"
        f"<span class=p>{s['proba']:.2f}</span></a>"
        for i, s in enumerate(sections))
    return f"<h3>Sections</h3><div class=toc>{links}</div>" if links else ""


def _stage(rows_html: str, rail_html: str) -> str:
    return (f"{LEGEND}<div class=stage><div class=doc>{rows_html}</div>"
            f"<div class=rail>{rail_html}</div></div>")


def _render_dossier(doc: dict) -> str:
    """The two-pane review for a paper dossier (stored or freshly audited)."""
    rows = "".join(
        _doc_row(i, s, highlight_html(s["excerpt"], s.get("hl_levels")) + "...")
        for i, s in enumerate(doc["sections"]))
    hot = doc["n_flagged"] > 0
    badge = ("<span class='badge flag'>FLAGGED</span>" if hot
             else "<span class='badge ok'>clear</span>")
    pills = _drive_pills(doc.get("top_tells", []))
    rail = (f"<div id=ov>{_dial(doc['max_proba'], hot)}"
            f"<div class=muted>max P(LLM) &middot; mean {doc['mean_proba']:.2f}</div>"
            f"<div style='margin-top:5px'>{badge} <span class=muted>"
            f"{doc['n_flagged']} of {doc['n_sections']} sections</span></div></div>"
            f"<h3>Tells driving this paper</h3>"
            f"<div>{pills or '<span class=muted>none</span>'}</div>"
            f"{_toc(doc['sections'])}")
    return _stage(rows, rail)


def _warming() -> str:
    return _page("LLM Tell Auditor", "<h1>LLM Tell Auditor</h1><p class=sub>Warming up...</p>")


# --- feature-store refresh + engine warmup -------------------------------
def _refresh() -> None:
    import hopsworks
    fs = hopsworks.login().get_feature_store()
    df = fs.get_feature_group("paper_dossiers", version=1).read()
    rows = df.to_dict("records") if not df.empty else []
    rows.sort(key=lambda r: (r.get("flagged_share", 0), r.get("max_proba", 0)), reverse=True)
    DOSSIERS["rows"] = rows
    DOSSIERS["by_id"] = {r["paper_id"]: r for r in rows}
    print(f"refresh: {len(rows)} dossiers", flush=True)


async def refresh_loop():
    while True:
        try:
            await asyncio.to_thread(_refresh)
        except Exception as e:
            print(f"refresh failed: {e}", flush=True)
        await asyncio.sleep(REFRESH_S)


def _warmup() -> None:
    import hopsworks
    proj = hopsworks.login()
    mdir = proj.get_model_registry().get_model("tell_classifier", version=A.MODEL_VERSION).download()
    ENGINE["auditor"] = A.load_auditor(mdir)
    try:
        os.environ["ANTHROPIC_API_KEY"] = hopsworks.get_secrets_api().get_secret("ANTHROPIC_API_KEY").value
        ENGINE["client"] = anthropic.Anthropic()
    except Exception as e:
        ENGINE["note"] = "live scoring is on, but no Anthropic key so written feedback is off"
        print(f"warmup: no anthropic client: {e}", flush=True)
    ENGINE["ready"] = ENGINE["auditor"] is not None
    print(f"warmup: engine ready={ENGINE['ready']} client={ENGINE['client'] is not None}", flush=True)


async def warmup_task():
    try:
        await asyncio.to_thread(_warmup)
    except Exception as e:
        ENGINE["note"] = f"live auditor failed to load: {str(e)[:160]}"
        print(f"warmup failed: {e}", flush=True)


@asynccontextmanager
async def _lifespan(_):
    tasks = [asyncio.create_task(refresh_loop()), asyncio.create_task(warmup_task())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(lifespan=_lifespan)
_PROXY_MOUNT = re.compile(r"^/hopsworks-api/pythonapp/[^/]+/[^/]+")


class StripForwardedPrefix:
    """Strip the Hopsworks proxy mount from the path (no APP_BASE_URL_PATH env /
    X-Forwarded-Prefix header here) so the app's routes match; record it as
    root_path for absolute links. The readiness probe on / passes through."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            prefix = dict(scope.get("headers") or {}).get(b"x-forwarded-prefix", b"").decode().rstrip("/")
            if not prefix:
                m = _PROXY_MOUNT.match(scope["path"])
                prefix = m.group(0) if m else ""
            if prefix and scope["path"].startswith(prefix):
                scope = dict(scope)
                scope["path"] = scope["path"][len(prefix):] or "/"
                scope["root_path"] = prefix
        await self.inner(scope, receive, send)


application = StripForwardedPrefix(app)


# --- spotlight: click a locatable tell pill to isolate its marks ----------
SPOT_SCRIPT = """<script>
document.addEventListener('click',function(e){
 var p=e.target.closest?e.target.closest('.pill[data-spot]'):null; if(!p)return;
 var cat=p.getAttribute('data-spot'), docs=document.querySelectorAll('.doc');
 if(!docs.length)return;
 var on=docs[0].getAttribute('data-spot')!==cat;
 docs.forEach(function(d){on?d.setAttribute('data-spot',cat):d.removeAttribute('data-spot')});
 document.querySelectorAll('.pill.on').forEach(function(x){x.classList.remove('on')});
 if(on)p.classList.add('on');
});
</script>"""


# --- the streaming client (inline, progressive enhancement) --------------
def _script(base: str) -> str:
    return "<script>" + STREAM_JS.replace("__BASE__", base) + "</script>"


STREAM_JS = r"""
const BASE="__BASE__";
const el=t=>document.createElement(t);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const barHTML=(p,hot)=>`<span class="bar${hot?' hot':''}"><i style="width:${Math.max(2,Math.round(p*100))}%"></i></span>`;
const C=2*Math.PI*50;
const dialHTML=(p,hot)=>`<div class=dial><svg width=110 height=110 viewBox="0 0 110 110">`+
 `<circle cx=55 cy=55 r=50 fill=none stroke="#E3E7EC" stroke-width=10 />`+
 `<circle cx=55 cy=55 r=50 fill=none stroke="${hot?'#C0394B':'#0E9A73'}" stroke-width=10 `+
 `stroke-linecap=round stroke-dasharray=${C.toFixed(1)} stroke-dashoffset=${(C*(1-Math.min(1,p))).toFixed(1)} />`+
 `</svg><div class=n>${p.toFixed(2)}</div></div>`;
const af=document.getElementById('af'), q=document.getElementById('q'), go=document.getElementById('go');
if(af && window.fetch && window.ReadableStream){
 af.addEventListener('submit', async e=>{
  e.preventDefault();
  const text=q.value.trim(); if(!text) return;
  go.disabled=true; go.textContent='Reviewing...';
  let res=document.getElementById('result');
  if(!res){res=el('div'); res.id='result'; af.after(res);} res.innerHTML='';
  const head=el('div'); res.appendChild(head);
  const stage=el('div'); stage.className='stage'; res.appendChild(stage);
  const doc=el('div'); doc.className='doc scanning'; stage.appendChild(doc);
  const rail=el('div'); rail.className='rail'; stage.appendChild(rail);
  const ov=el('div'); ov.id='ov'; rail.appendChild(ov);
  const status=el('div'); status.id='progress'; status.textContent='Reading...'; rail.appendChild(status);
  const toc=el('div'); toc.className='toc'; rail.appendChild(toc);

  // network state, filled by the reader; the renderer below paces the visuals so
  // the "flow" looks right whether the proxy trickles the stream or buffers it.
  const queue=[]; let streaming=true, label='Review';
  let overallMsg=null, noteMsg=null, errMsg=null, redirectUrl=null, fbText='', fbStarted=false, fbDone=false;

  (async ()=>{
   try{
    const r=await fetch(BASE+'/audit/stream',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'q='+encodeURIComponent(text)});
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    for(;;){const {value,done}=await rd.read(); if(done)break; buf+=dec.decode(value,{stream:true});
     let nl; while((nl=buf.indexOf('\n'))>=0){const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1); if(!line)continue;
      const m=JSON.parse(line);
      if(m.type==='redirect') redirectUrl=m.url;
      else if(m.type==='meta'){label=m.label; head.innerHTML=`<h2>${label}</h2>`;}
      else if(m.type==='item') queue.push(m.item);
      else if(m.type==='overall') overallMsg=m;
      else if(m.type==='fb_start') fbStarted=true;
      else if(m.type==='fb') fbText+=m.t;
      else if(m.type==='note') noteMsg=m.t;
      else if(m.type==='error') errMsg=m.msg;
     }
    }
   }catch(err){errMsg='Stream failed: '+err;}
   streaming=false; fbDone=true;
  })();

  let n=0;
  function addRow(it){
   const row=el('div'); row.className='drow'+(it.flagged?' flagged':''); row.id='s'+n;
   if(it.title && !it.title.startsWith('Paragraph')){
    const tt=el('div'); tt.className='dtitle'; tt.textContent=it.title; row.appendChild(tt);
    const a=el('a'); a.href='#s'+n; a.className=it.flagged?'f':'';
    a.innerHTML=`<span>${it.title.slice(0,34)}</span><span class=p>${it.proba.toFixed(2)}</span>`;
    toc.appendChild(a);
   }
   const tx=el('div'); tx.className='dtext'; tx.innerHTML=it.html;
   const nt=el('div'); nt.className='dnote'+(it.flagged?' flagged':'');
   nt.innerHTML=`<span class=p>${barHTML(it.proba,it.flagged)} ${it.proba.toFixed(2)}</span><span>${it.comment}</span>`;
   row.appendChild(tx); row.appendChild(nt); doc.appendChild(row);
  }

  // paced renderer: one passage every ~150ms while the scan sweeps
  while(streaming || queue.length){
   if(redirectUrl){location.href=redirectUrl; return;}
   if(queue.length){ addRow(queue.shift()); n++; status.textContent=`Checked ${n} passage${n>1?'s':''}...`; await sleep(150); }
   else await sleep(40);
  }
  if(redirectUrl){location.href=redirectUrl; return;}
  if(errMsg){ status.textContent=errMsg; doc.classList.remove('scanning'); go.disabled=false; go.textContent='Audit'; return; }

  doc.classList.remove('scanning'); status.textContent='';
  if(overallMsg){ const m=overallMsg;
   ov.innerHTML=dialHTML(m.proba,m.flagged)+
    `<div class=muted>P(matches LLM style)</div>`+
    `<div style="margin-top:5px"><span class="badge ${m.flagged?'flag':(m.short?'warn':'ok')}">`+
    `${m.flagged?'FLAGGED':(m.short?'too short to trust':'clear')}</span></div>`; }
  if(noteMsg){ const p=el('div'); p.className='muted'; p.style.fontSize='.85rem'; p.style.margin='8px 0';
   p.textContent=noteMsg; rail.appendChild(p); }

  if(fbStarted){
   const h=el('h3'); h.textContent='Review'; rail.appendChild(h);
   const box=el('div'); box.className='review cursor'; rail.appendChild(box);
   // typewriter the (possibly already fully received) review, client-side paced
   let i=0;
   while(!fbDone || i<fbText.length){
    if(i<fbText.length){ box.textContent+=fbText.slice(i,i+2); i+=2; await sleep(10); }
    else await sleep(30);
   }
   box.classList.remove('cursor');
  }
  go.disabled=false; go.textContent='Audit';
 });
}
"""


# --- routes --------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "dossiers": len(DOSSIERS["rows"]), "engine": ENGINE["ready"]}


_SORTS = {"flagged": lambda r: (r.get("flagged_share", 0), r.get("max_proba", 0)),
          "max": lambda r: r.get("max_proba", 0),
          "mean": lambda r: r.get("mean_proba", 0)}


def _qs(cat: str, sort: str) -> str:
    parts = [p for p in (f"cat={cat}" if cat else "", f"sort={sort}" if sort != "flagged" else "") if p]
    return "?" + "&".join(parts) if parts else ""


@app.get("/", response_class=HTMLResponse)
def index(request: Request, cat: str = "", sort: str = "flagged"):
    base = _base(request)
    sort = sort if sort in _SORTS else "flagged"
    rows = DOSSIERS["rows"]
    note = f"<div class='band'><b>Note:</b> {_e(ENGINE['note'])}</div>" if ENGINE["note"] else ""

    tiles, cats_html = "", ""
    shown = rows
    if rows:
        n_flagged_papers = sum(1 for r in rows if r.get("n_flagged", 0) > 0)
        cats = sorted({r.get("category", "") for r in rows if r.get("category")})
        tiles = ("<div class=tiles>"
                 f"<div class=tile><div class=v>{len(rows)}</div><div class=l>papers audited</div></div>"
                 f"<div class=tile><div class=v>{sum(r.get('n_sections', 0) for r in rows)}</div>"
                 "<div class=l>sections scored</div></div>"
                 f"<div class=tile><div class=v>{n_flagged_papers}</div>"
                 "<div class=l>papers with a flagged section</div></div>"
                 f"<div class=tile><div class=v>{len(cats)}</div><div class=l>arXiv categories</div></div>"
                 "</div>")
        shown = [r for r in rows if not cat or r.get("category") == cat]
        shown.sort(key=_SORTS[sort], reverse=True)
        cat_links = f"<a href='{base}/{_qs('', sort)}' class='{'on' if not cat else ''}'>all</a>" + "".join(
            f"<a href='{base}/{_qs(c, sort)}' class='{'on' if c == cat else ''}'>{_e(c)}</a>" for c in cats)
        cats_html = f"<div class=cats>{cat_links}</div>"

    def th(label: str, key: str) -> str:
        return (f"<th class=num><a href='{base}/{_qs(cat, key)}' "
                f"class='{'on' if sort == key else ''}'>{label}</a></th>")

    trs = "".join(
        f"<tr><td><a href='{base}/paper/{_e(r['paper_id'])}'>{_e((r.get('title') or r['paper_id'])[:80])}</a>"
        f"<div class=muted style='font-size:.8rem'>{_e(r['paper_id'])} &middot; {_e(r.get('category', ''))}</div></td>"
        f"<td class=num>{r.get('n_flagged', 0)}/{r.get('n_sections', 0)}</td>"
        f"<td class=num>{_bar(r.get('max_proba', 0))} {r.get('max_proba', 0):.2f}</td>"
        f"<td class=num>{r.get('mean_proba', 0):.2f}</td></tr>" for r in shown)
    table = (f"<table><thead><tr><th>Paper</th>{th('Flagged', 'flagged')}"
             f"{th('Max P(LLM)', 'max')}{th('Mean', 'mean')}</tr></thead><tbody>"
             + trs + "</tbody></table>") if shown else "<p class=muted>No dossiers yet.</p>"
    body = (
        "<h1>LLM Tell Auditor</h1>"
        "<p class=sub>Paste anything to audit its writing style, or browse recent "
        f"arXiv preprints already audited.</p>{note}{tiles}"
        f"{FORM.format(base=base)}{LEGEND}{BANNER}"
        f"<div id=result></div>"
        f"<h2>Audited preprints ({len(shown)})</h2>{cats_html}{table}"
        "<p class=foot>Flagged = P(LLM) &ge; 0.5 from the tell_classifier, a calibrated "
        "model over 16 stylometric tells, held out by paper. Matches known tells, nothing more.</p>")
    return HTMLResponse(_page("LLM Tell Auditor", body, _script(base),
                              desc="Audit academic prose for LLM writing tells. Signal, not verdict."))


def _ndjson(obj) -> str:
    return json.dumps(obj) + "\n"


@app.post("/audit/stream")
def audit_stream(request: Request, q: str = Form("")):
    base = _base(request)
    q = (q or "").strip()[:MAX_CHARS]

    def gen():
        if not ENGINE["ready"]:
            yield _ndjson({"type": "error", "msg": "Warming up, try again in a moment."})
            return
        if not q:
            yield _ndjson({"type": "error", "msg": "Nothing to audit."})
            return
        aid = A.extract_arxiv_id(q)
        if aid and (aid in DOSSIERS["by_id"] or f"{aid}v1" in DOSSIERS["by_id"]):
            key = aid if aid in DOSSIERS["by_id"] else f"{aid}v1"
            yield _ndjson({"type": "redirect", "url": f"{base}/paper/{key}"})
            return

        items, probas = [], []
        if aid:
            yield _ndjson({"type": "meta", "label": f"Live audit of {aid}"})
            for sec in A.iter_paper_sections(aid, ENGINE["auditor"]):
                sec["html"] = highlight_html(sec["excerpt"], sec.get("hl_levels")) + "..."
                sec["comment"] = _comment(sec)
                items.append(sec)
                probas.append(sec["proba"])
                yield _ndjson({"type": "item", "item": sec})
                time.sleep(0.12)  # pace the sweep so the review reads as a flow
            if not items:
                yield _ndjson({"type": "error", "msg": f"No LaTeX source or scorable section for {aid}."})
                return
            flagged = sum(1 for p in probas if p >= A.FLAG_THRESHOLD)
            yield _ndjson({"type": "overall", "proba": max(probas), "flagged": flagged > 0, "short": False})
            yield _ndjson({"type": "note", "t": f"{flagged} of {len(items)} sections flagged. "
                          "Per-section evidence above; open the paper on arXiv to read in full."})
            yield _ndjson({"type": "done"})
            return

        # raw prose: paragraph by paragraph, then streamed feedback
        yield _ndjson({"type": "meta", "label": "Reviewing your text"})
        for i, para in enumerate(A.split_paragraphs(q)):
            it = A.score_item(f"Paragraph {i + 1}", para, ENGINE["auditor"])
            it["html"] = highlight_html(para, it.get("hl_levels"))  # the FULL passage, marked
            it["comment"] = _comment(it)
            items.append(it)
            probas.append(it["proba"])
            yield _ndjson({"type": "item", "item": it})
            time.sleep(0.12)  # pace the sweep so the review reads as a flow
        res = A.audit_text(q, ENGINE["auditor"])
        yield _ndjson({"type": "overall", "proba": res["proba"],
                       "flagged": res["flagged"], "short": res["short"]})
        if ENGINE["client"]:
            yield _ndjson({"type": "fb_start"})
            try:
                for delta in E.explain_stream(q, res, ENGINE["client"]):
                    yield _ndjson({"type": "fb", "t": delta})
            except Exception as ex:
                yield _ndjson({"type": "fb", "t": f"\n[feedback unavailable: {str(ex)[:120]}]"})
        else:
            yield _ndjson({"type": "note", "t": ENGINE["note"] or "Written feedback is off."})
        yield _ndjson({"type": "done"})

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/audit", response_class=HTMLResponse)
def audit_submit(request: Request, q: str = Form("")):
    """No-JS fallback: one server-rendered result page, same two-pane review."""
    base = _base(request)
    q = (q or "").strip()
    if not ENGINE["ready"]:
        return HTMLResponse(_warming(), status_code=503)
    if not q:
        return RedirectResponse(f"{base}/", status_code=303)

    aid = A.extract_arxiv_id(q)
    if aid:
        if aid in DOSSIERS["by_id"] or f"{aid}v1" in DOSSIERS["by_id"]:
            key = aid if aid in DOSSIERS["by_id"] else f"{aid}v1"
            return RedirectResponse(f"{base}/paper/{key}", status_code=303)
        dossier = A.audit_paper(aid, ENGINE["auditor"])
        if not dossier:
            return HTMLResponse(_page("Not found",
                f"<p><a href='{base}/'>&larr; back</a></p><h1>Could not audit {_e(aid)}</h1>"
                "<p class=muted>No LaTeX source, or no section long enough to score.</p>"), status_code=404)
        body = (f"<p><a href='{base}/'>&larr; audit another</a></p><h1>{_e(dossier['title'])}</h1>"
                f"<p class=sub>{_e(aid)} &middot; live audit &middot; "
                f"<a href='https://arxiv.org/abs/{_e(aid)}'>arXiv</a></p>{BANNER}"
                f"{_render_dossier(dossier)}")
        return HTMLResponse(_page(dossier["title"], body,
                                  desc=f"{dossier['n_flagged']} of {dossier['n_sections']} sections flagged."))

    text = q[:MAX_CHARS]
    paras = A.split_paragraphs(text)
    items = [A.score_item(f"Paragraph {i + 1}", p, ENGINE["auditor"]) for i, p in enumerate(paras)]
    rows = "".join(_doc_row(i, it, highlight_html(p, it.get("hl_levels")))
                   for i, (it, p) in enumerate(zip(items, paras)))
    res = A.audit_text(text, ENGINE["auditor"])
    badge = ("<span class='badge flag'>FLAGGED</span>" if res["flagged"]
             else "<span class='badge ok'>clear</span>")
    short = " <span class='badge warn'>too short to trust</span>" if res["short"] else ""
    if ENGINE["client"]:
        try:
            feedback = f"<h3>Review</h3><div class=review>{_e(E.explain(text, res, ENGINE['client']))}</div>"
        except Exception as ex:
            feedback = f"<p class=muted>Feedback unavailable: {_e(str(ex)[:140])}</p>"
    else:
        feedback = f"<p class=muted>{_e(ENGINE['note'] or 'Written feedback is off.')}</p>"
    fired = _tell_pills(res["top_tells"])
    human = _tell_pills(sorted((t for t in res["all_tells"] if t["contribution"] < 0),
                               key=lambda t: t["contribution"])[:4])
    rail = (f"<div id=ov>{_dial(res['proba'], res['flagged'])}"
            f"<div class=muted>P(matches LLM style) &middot; {res['n_words']} words</div>"
            f"<div style='margin-top:5px'>{badge}{short}</div></div>"
            f"{feedback}"
            f"<h3>Toward LLM style</h3><div>{fired or '<span class=muted>none</span>'}</div>"
            f"<h3>Toward human style</h3><div>{human or '<span class=muted>none</span>'}</div>")
    body = (
        f"<p><a href='{base}/'>&larr; audit another</a></p><h1>Style audit</h1>"
        f"{BANNER}{_stage(rows, rail)}"
        "<p class=foot>Contribution is each tell's push on the log-odds "
        "(+ toward LLM style, - toward human style); the value is the measured feature.</p>")
    return HTMLResponse(_page("Style audit", body, desc="Stylometric audit of pasted prose."))


@app.get("/paper/{paper_id}", response_class=HTMLResponse)
def paper(request: Request, paper_id: str):
    if not DOSSIERS["rows"]:
        return HTMLResponse(_warming())
    r = DOSSIERS["by_id"].get(paper_id)
    base = _base(request)
    if r is None:
        return HTMLResponse(_page("Not found",
                            f"<h1>Not found</h1><p><a href='{base}/'>&larr; all papers</a></p>"
                            f"<p class=muted>No dossier for {_e(paper_id)}.</p>"), status_code=404)
    # sections_json holds sections + top_tells; the paper-level stats are row columns
    doc = json.loads(r["sections_json"])
    for k in ("n_flagged", "n_sections", "mean_proba", "max_proba"):
        doc.setdefault(k, r.get(k, 0))
    body = (
        f"<p><a href='{base}/'>&larr; all papers</a></p><h1>{_e(r.get('title') or paper_id)}</h1>"
        f"<p class=sub>{_e(paper_id)} &middot; {_e(r.get('category', ''))} &middot; "
        f"<a href='https://arxiv.org/abs/{_e(paper_id)}'>arXiv</a></p>{BANNER}"
        f"{_render_dossier(doc)}")
    return HTMLResponse(_page(r.get("title") or paper_id, body,
                              desc=f"{r.get('n_flagged', 0)} of {r.get('n_sections', 0)} sections flagged."))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="0.0.0.0", port=int(os.environ.get("APP_PORT", 8000)))
