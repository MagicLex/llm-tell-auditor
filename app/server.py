"""LLM Tell Auditor -- server-rendered viewer + streaming live auditor (no SPA).

Base experience is fully server-rendered and JS-free: browse precomputed
`paper_dossiers`, and a plain `POST /audit` returns a complete result page
(crawlable, works with JS off). On top of that, a progressive enhancement: if JS
is on, the form streams instead, scoring item by item (sections for a paper,
paragraphs for pasted text) with cards that slide in, stabilo-style highlights on
the flagged words, and the plain-language feedback typed in as the LLM writes it.

Signal, not verdict, enforced in the explanation prompt. One tell family
(stylometric polish). Only token-level tells are highlighted; distributional
ones have no single locus and are left unmarked on purpose.
"""
import asyncio
import html
import json
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
*{box-sizing:border-box} body{margin:0;background:#0E1117;color:#E6E8EB;
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
a{color:#4Fd1a8;text-decoration:none} a:hover{text-decoration:underline}
.wrap{max-width:960px;margin:0 auto;padding:24px 18px 64px}
h1{font-size:1.5rem;margin:0 0 2px} h2{font-size:1.05rem;margin:24px 0 10px}
.sub{color:#8B93A0;margin:0 0 18px}
.band{border-left:5px solid #1EB182;background:#141A22;padding:12px 16px;
 border-radius:6px;margin:0 0 20px;font-size:.92rem;color:#C7CEd8} .band b{color:#E6E8EB}
form.audit{margin:0 0 14px} textarea{width:100%;min-height:120px;background:#0B0F15;
 color:#E6E8EB;border:1px solid #2A333F;border-radius:8px;padding:12px;font:inherit;resize:vertical}
.row{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
button{background:#1EB182;color:#08120D;border:none;font-weight:700;padding:9px 18px;
 border-radius:8px;cursor:pointer;font:inherit} button:hover{background:#28c493}
button:disabled{opacity:.5;cursor:default}
.hint{color:#8B93A0;font-size:.85rem}
table{width:100%;border-collapse:collapse;font-size:.92rem}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #222A34}
th{color:#8B93A0;font-weight:600} tr:hover td{background:#141A22}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{display:inline-block;height:8px;border-radius:4px;vertical-align:middle;transition:width .5s ease}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.78rem;font-weight:600}
.flag{background:#3a1d1d;color:#f0a4a4} .ok{background:#16281f;color:#7fd6ab}
.warn{background:#2e2716;color:#e0c879}
.pill{display:inline-block;background:#1A2029;border:1px solid #2A333F;color:#B9c2cE;
 padding:1px 8px;margin:2px 4px 2px 0;border-radius:10px;font-size:.8rem}
.contrib{color:#e0a0a0} .neg{color:#8fb0d0} .muted{color:#8B93A0}
.sec{border:1px solid #222A34;border-radius:8px;padding:14px 16px;margin:12px 0;background:#11161D}
.sec.flagged{border-color:#4a2a2a} .sec h3{margin:0 0 6px;font-size:1rem}
.exc{color:#c3ccd6;font-size:.9rem;margin-top:8px;border-left:2px solid #2A333F;padding-left:10px}
.feedback{background:#101a16;border:1px solid #1d3a2e;border-radius:10px;padding:16px 18px;
 margin:12px 0;font-size:.98rem;line-height:1.6;white-space:pre-wrap}
.score{font-size:2.2rem;font-weight:800;font-variant-numeric:tabular-nums}
.foot{color:#6b7480;font-size:.82rem;margin-top:36px;border-top:1px solid #222A34;padding-top:14px}
/* stabilo-style highlights: translucent marker over the flagged token */
/* stabilo highlights, layered by signal strength: l3 marker, l2 wash, l1 underline */
mark[class^=hl-]{padding:0 .12em;border-radius:3px;box-decoration-break:clone;
 -webkit-box-decoration-break:clone;background:none;color:inherit}
.l3{color:#0c0f13;font-weight:600}
.hl-transition.l3{background:#ffe14d} .hl-transition.l2{background:rgba(255,225,77,.30)} .hl-transition.l1{border-bottom:2px dotted #ffe14d}
.hl-booster.l3{background:#ff9de0} .hl-booster.l2{background:rgba(255,157,224,.30)} .hl-booster.l1{border-bottom:2px dotted #ff9de0}
.hl-hedge.l3{background:#8affc1} .hl-hedge.l2{background:rgba(138,255,193,.28)} .hl-hedge.l1{border-bottom:2px dotted #8affc1}
.hl-dash.l3{background:#8fd0ff} .hl-dash.l2{background:rgba(143,208,255,.30)} .hl-dash.l1{border-bottom:2px dotted #8fd0ff}
.hl-punc.l3{background:#ffc07a} .hl-punc.l2{background:rgba(255,192,122,.30)} .hl-punc.l1{border-bottom:2px dotted #ffc07a}
.legend{font-size:.8rem;color:#8B93A0;margin:6px 0 2px} .legend mark{margin-right:2px}
/* streaming polish */
#progress{color:#8B93A0;font-size:.85rem;margin:8px 0;min-height:1.2em}
.cursor::after{content:"\\258c";color:#1EB182;animation:blink 1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
/* two-pane review: document left, sticky score+review rail right, scan sweep */
.stage{display:grid;grid-template-columns:1fr 300px;gap:18px;align-items:start;margin-top:10px}
.doc{position:relative;overflow:hidden;border:1px solid #222A34;border-radius:10px;background:#0f141b}
.doc.scanning::after{content:"";position:absolute;left:0;right:0;top:-100px;height:100px;pointer-events:none;
 background:linear-gradient(180deg,transparent,rgba(30,177,130,.18),transparent);animation:scan 1.9s linear infinite}
@keyframes scan{from{top:-100px}to{top:100%}}
.drow{padding:13px 15px;border-bottom:1px solid #171d25;border-left:3px solid transparent}
.drow:last-child{border-bottom:none} .drow.flagged{border-left-color:#c8565f}
.dtext{line-height:1.85;font-size:.97rem;color:#e2e7ec}
.dnote{margin-top:7px;font-size:.82rem;color:#9aa4b0;display:flex;gap:8px;align-items:baseline}
.dnote.flagged{color:#e3b7b7} .dnote .p{font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}
.rail{position:sticky;top:14px;align-self:start} #ov{margin:0 0 8px}
.rail .score{font-size:2rem;font-weight:800;font-variant-numeric:tabular-nums}
.review{background:#101a16;border:1px solid #1d3a2e;border-radius:10px;padding:14px 16px;margin:6px 0;
 white-space:pre-wrap;line-height:1.6;font-size:.92rem}
@media(max-width:760px){.stage{grid-template-columns:1fr}.rail{position:static}}
"""

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


def _bar(p: float) -> str:
    w = max(2, round(p * 120))
    hue = 120 - round(p * 120)
    return f"<span class=bar style='width:{w}px;background:hsl({hue},55%,45%)'></span>"


def _page(title: str, body: str, script: str = "") -> str:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{CSS}</style></head>"
            f"<body><div class=wrap>{body}</div>{script}</body></html>")


def _tell_pills(tells) -> str:
    return "".join(
        f"<span class=pill>{_e(t['tell'])} "
        f"<span class={'contrib' if t['contribution']>0 else 'neg'}>{t['contribution']:+.2f}</span> "
        f"<span class=muted>({t['value']})</span></span>"
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


def _section_html(s: dict) -> str:
    cls = "sec flagged" if s["flagged"] else "sec"
    badge = ("<span class='badge flag'>FLAGGED</span>" if s["flagged"]
             else "<span class='badge ok'>clear</span>")
    pills = _tell_pills(s.get("top_tells", []))
    return (f"<div class='{cls}'><h3>{_e(s['title'])} {badge}</h3>"
            f"<div class=muted>{_bar(s['proba'])} P(LLM) {s['proba']:.3f} &middot; {s['n_words']} words</div>"
            f"<div style='margin-top:8px'>{pills or '<span class=muted>no tell pushed toward LLM</span>'}</div>"
            f"<div class=exc>{highlight_html(s['excerpt'])}...</div></div>")


def _render_sections(doc: dict) -> str:
    paper_tells = "".join(
        f"<span class=pill>{_e(t['tell'])} <b class=contrib>+{t['drive']:.2f}</b></span>"
        for t in doc.get("top_tells", []))
    secs = "".join(_section_html(s) for s in doc["sections"])
    return (f"<h2>Tells driving this paper</h2><div>{paper_tells or '<span class=muted>none</span>'}</div>"
            f"{LEGEND}<h2>Per-section evidence</h2>{secs}")


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
    mdir = proj.get_model_registry().get_model("tell_classifier", version=1).download()
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


# --- the streaming client (inline, progressive enhancement) --------------
def _script(base: str) -> str:
    return "<script>" + STREAM_JS.replace("__BASE__", base) + "</script>"


STREAM_JS = r"""
const BASE="__BASE__";
const el=t=>document.createElement(t);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const barHTML=p=>`<span class="bar" style="width:${Math.max(2,Math.round(p*120))}px;background:hsl(${120-Math.round(p*120)},55%,45%)"></span>`;
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

  function addRow(it){
   const row=el('div'); row.className='drow'+(it.flagged?' flagged':'');
   const tx=el('div'); tx.className='dtext'; tx.innerHTML=it.html;
   const nt=el('div'); nt.className='dnote'+(it.flagged?' flagged':'');
   nt.innerHTML=`<span class=p>${barHTML(it.proba)} ${it.proba.toFixed(2)}</span><span>${it.comment}</span>`;
   row.appendChild(tx); row.appendChild(nt); doc.appendChild(row);
  }

  // paced renderer: one passage every ~150ms while the scan sweeps
  let n=0;
  while(streaming || queue.length){
   if(redirectUrl){location.href=redirectUrl; return;}
   if(queue.length){ addRow(queue.shift()); n++; status.textContent=`Checked ${n} passage${n>1?'s':''}...`; await sleep(150); }
   else await sleep(40);
  }
  if(redirectUrl){location.href=redirectUrl; return;}
  if(errMsg){ status.textContent=errMsg; doc.classList.remove('scanning'); go.disabled=false; go.textContent='Audit'; return; }

  doc.classList.remove('scanning'); status.textContent='';
  if(overallMsg){ const m=overallMsg;
   ov.innerHTML=`<span class="score">${m.proba.toFixed(2)}</span>`+
    `<div class=muted>P(matches LLM style)</div>`+
    `<div style="margin-top:5px"><span class="badge ${m.flagged?'flag':(m.short?'warn':'ok')}">`+
    `${m.flagged?'FLAGGED':(m.short?'too short to trust':'clear')}</span></div>`; }
  if(noteMsg){ const p=el('div'); p.className='muted'; p.style.fontSize='.85rem'; p.style.margin='8px 0';
   p.textContent=noteMsg; rail.appendChild(p); }

  if(fbStarted){
   const h=el('h3'); h.textContent='Review'; h.style.margin='14px 0 6px'; rail.appendChild(h);
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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    base = _base(request)
    rows = DOSSIERS["rows"]
    note = f"<div class='band'><b>Note:</b> {_e(ENGINE['note'])}</div>" if ENGINE["note"] else ""
    trs = "".join(
        f"<tr><td><a href='{base}/paper/{_e(r['paper_id'])}'>{_e((r.get('title') or r['paper_id'])[:80])}</a>"
        f"<div class=muted style='font-size:.8rem'>{_e(r['paper_id'])} &middot; {_e(r.get('category',''))}</div></td>"
        f"<td class=num>{r.get('n_flagged',0)}/{r.get('n_sections',0)}</td>"
        f"<td class=num>{_bar(r.get('max_proba',0))} {r.get('max_proba',0):.2f}</td>"
        f"<td class=num>{r.get('mean_proba',0):.2f}</td></tr>" for r in rows)
    table = ("<table><thead><tr><th>Paper</th><th class=num>Flagged</th>"
             "<th class=num>Max P(LLM)</th><th class=num>Mean</th></tr></thead><tbody>"
             + trs + "</tbody></table>") if rows else "<p class=muted>No dossiers yet.</p>"
    body = (
        "<h1>LLM Tell Auditor</h1>"
        "<p class=sub>Paste anything to audit its writing style, or browse recent "
        f"arXiv preprints already audited.</p>{note}"
        f"{FORM.format(base=base)}{LEGEND}{BANNER}"
        f"<div id=result></div>"
        f"<h2>Audited preprints ({len(rows)})</h2>{table}"
        "<p class=foot>Flagged = P(LLM) &ge; 0.5 from tell_classifier v1, a calibrated "
        "logistic over 16 stylometric tells, held out by paper. Matches known tells, nothing more.</p>")
    return HTMLResponse(_page("LLM Tell Auditor", body, _script(base)))


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
    """No-JS fallback: one server-rendered result page."""
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
                f"<p>{dossier['n_flagged']} of {dossier['n_sections']} sections flagged "
                f"&middot; mean P(LLM) {dossier['mean_proba']:.3f} &middot; max {dossier['max_proba']:.3f}</p>"
                f"{_render_sections(dossier)}")
        return HTMLResponse(_page(dossier["title"], body))

    text = q[:MAX_CHARS]
    res = A.audit_text(text, ENGINE["auditor"])
    badge = ("<span class='badge flag'>FLAGGED</span>" if res["flagged"]
             else "<span class='badge ok'>clear</span>")
    short = "<span class='badge warn'>too short to trust</span>" if res["short"] else ""
    if ENGINE["client"]:
        try:
            feedback = f"<div class=feedback>{_e(E.explain(text, res, ENGINE['client']))}</div>"
        except Exception as ex:
            feedback = f"<p class=muted>Feedback unavailable: {_e(str(ex)[:140])}</p>"
    else:
        feedback = f"<p class=muted>{_e(ENGINE['note'] or 'Written feedback is off.')}</p>"
    fired = _tell_pills(res["top_tells"])
    human = _tell_pills(sorted((t for t in res["all_tells"] if t["contribution"] < 0),
                               key=lambda t: t["contribution"])[:4])
    body = (
        f"<p><a href='{base}/'>&larr; audit another</a></p><h1>Style audit</h1>"
        f"<p><span class=score>{res['proba']:.2f}</span> <span class=muted>P(matches LLM style)</span> "
        f"{badge} {short} <span class=muted>&middot; {res['n_words']} words</span></p>"
        f"<div>{_bar(res['proba'])}</div>{BANNER}"
        f"<h2>What this means</h2>{feedback}"
        f"<h2>Tells pushing toward LLM style</h2><div>{fired or '<span class=muted>none</span>'}</div>"
        f"<h2>Tells pushing toward human style</h2><div>{human or '<span class=muted>none</span>'}</div>"
        f"{LEGEND}<h2>Your text</h2><div class=exc>{highlight_html(res['excerpt'])}"
        f"{'...' if len(text) > 400 else ''}</div>"
        "<p class=foot>Contribution is each tell's push on the log-odds "
        "(+ toward LLM style, - toward human style); the value is the measured feature.</p>")
    return HTMLResponse(_page("Style audit", body))


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
    doc = json.loads(r["sections_json"])
    body = (
        f"<p><a href='{base}/'>&larr; all papers</a></p><h1>{_e(r.get('title') or paper_id)}</h1>"
        f"<p class=sub>{_e(paper_id)} &middot; {_e(r.get('category',''))} &middot; "
        f"<a href='https://arxiv.org/abs/{_e(paper_id)}'>arXiv</a></p>{BANNER}"
        f"<p>{r.get('n_flagged',0)} of {r.get('n_sections',0)} sections flagged "
        f"&middot; mean P(LLM) {r.get('mean_proba',0):.3f} &middot; max {r.get('max_proba',0):.3f}</p>"
        f"{_render_sections(doc)}")
    return HTMLResponse(_page(r.get("title") or paper_id, body))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="0.0.0.0", port=int(os.environ.get("APP_PORT", 8000)))
