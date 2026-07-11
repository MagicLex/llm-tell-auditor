# #010: LLM Tell Auditor (arXiv preprints)

**Not a "BS detector". Not an "AI detector".** A transparent auditor of LLM writing tells.
Per paper: a dossier of which known LLM signatures fired, with citations in context and counts.
**Signal, not verdict. Evidence, not accusation.**

## The trap we refuse (read first)

Detecting LLM **style** is not detecting **bad science**. Vocabulary and embeddings catch AI writing,
not fraud or bad method. A great paper can be LLM-polished; garbage can be handwritten. If we ever
title this "real vs BS", we are measuring the dial (style) and pretending we measured the thing
(quality). We do not. The honest claim is only ever: *"matches known LLM writing tells."*

## Two phenomena, kept separate (never one bucket)

1. **Polish tells.** Em-dash rate, transition words (moreover/furthermore/notably), "not X but Y",
   "it's not X, it's Y", hedges, uniform sentence rhythm. Author fluent, LLM smoothed the prose.
2. **Tortured phrases.** "counterfeit consciousness" = AI. Comes from paraphrase tools dodging
   plagiarism detectors (paper mills), predates ChatGPT. Obfuscation, not polish. Different animal.

Reported as separate feature families. Never mixed into one score.

## AI-system card

| field | value |
|---|---|
| Problem type | Binary classification (per section: human-written vs LLM-authored), calibrated probability + evidence |
| ML-system type | **Agentic** (discovery + evidence + evolving ruleset), classifier at the core |
| KPI | Precision-in-the-wild: share of flagged tells a human reviewer confirms genuinely present |
| ML proxy metric | AUROC / precision on held-out content-controlled rewrite-pairs |
| Data source | arXiv LaTeX e-print source (cs.LG, cs.CL, stat.ML, cs.AI). Free, no auth, global |
| Labels (v1) | **Rewrite-pairs only.** real section vs LLM-authored twin (content-controlled) |
| Consumption | Server-rendered app: per-paper dossier (tells fired, citations, counts). No SPA. |
| Monitoring | Log inputs + tells + scores; watch tell drift (arms race, known tells self-correct over time) |

## Data: proven (2026-07-11)

- arXiv API metadata + abstract: `https://export.arxiv.org/api/query`. 200, free.
- Full text: PDF (messy) and **LaTeX e-print source** (`https://arxiv.org/e-print/<id>`). Clean text, the mine.
- Tells greppable directly in source (verified: em-dash count, transition-word count per paper).
- Rate: arXiv asks ~3s between API requests; bulk via Kaggle metadata / S3 if volume grows.

## The rewrite-pair firewall (core F design)

Do **not** paraphrase the original (LLM anchors on human prose, gives a contaminated pair). Instead:

```
human section
  -> [reader agent]  extract CONTENT SKELETON: every claim, number, section order. JSON, ZERO prose
  -> [naive writer agent(s)]  sees ONLY skeleton, NEVER original sentences, writes fresh in LLM voice
  -> LLM twin  (pure LLM surface, same content)
```

Razor: skeleton must be **content-complete**. Too thin, writer invents different claims, content
drifts, topic confound returns. All substance, none of the wording. Panel = several writer models
so the classifier learns "LLM-ness", not one model's fingerprint.

---

## Pipelines (ordered, FTI)

### F: Feature pipeline (agentic). *no blocker*
Skills: `hops-data-sources`, `hops-features`, `hops-fg`, `hops-agent-job`
1. Ingest arXiv LaTeX source for cs.LG, cs.CL, stat.ML, cs.AI. Parse `.tex` into clean text per section.
   Into FG `arxiv_papers_raw` (paper_id, category, section, human_text, submitted_date).
2. **Reader agent** per section produces a content skeleton (JSON facts, no prose).
3. **Naive writer panel** (firewall) produces the LLM twin text per skeleton. Into FG `paper_twins` (paper_id, section, pair_id, llm_text, writer_model).
4. Tell extraction (MITs) on both human_text and llm_text: polish-tell features + tortured-phrase hits + embedding.
   Into FG `paper_tells` (pair_id, section, features, embedding, label {human=0, llm=1}, family={polish,tortured}).
   Backfill over history, then incremental schedule.

### T: Training pipeline. *blocked by F*
Skills: `hops-eda`, `hops-fv`, `hops-transformations`, `hops-train`
- EDA: leakage check (pair_id must not straddle train/test, split by pair, never by row).
- Feature view over `paper_tells`, MDTs (scaling, imputation) attached to the view.
- Train calibrated classifier (logreg / gradient-boosting) human-vs-llm on pairs.
- Eval: AUROC, precision/recall, calibration curve, tell importances. Save JSON + PNG.
- Register model + tell-importance report in model registry.

### I: Inference pipeline (agentic auditor). *blocked by T*
Skills: `hops-online-inference`, `hops-agent-deployment`, `hops-app`
- For a new paper: extract tells, score with model.
- **Discovery agent**: surface *new* candidate tells (embedding clustering, discriminative phrases,
  tortured-phrase mining) not yet in the ruleset. Ruleset grows, living detector.
- **Dossier assembly**: tells fired + citation-in-context + counts, per family. Proof, not a number.
  Into FG `paper_dossiers`. Log all inputs + scores.
- Server-rendered app: paper list + per-paper dossier view.

## Honest caveats (README from day one)
- Label noise: twin authored by a model panel, real preprints polished many ways. Never claim beyond "matches known tells".
- Moving target: a known tell, once public, gets corrected. Arms race. Ruleset must evolve; metrics decay.
- False-positive ethics: non-native English authors over-flagged by naive detectors. We publish evidence, not verdicts, for this reason.
