"""Self-contained HTML report -- RAGProbe's face.

Design intent: a *verification ledger*, not a dashboard. Paper-and-ink
palette, serif masthead, verdict stamps, proofreader-style wavy underlines
on flagged sentences, and a scorer x contamination-type detection matrix
as the centerpiece. Zero external JS; one Google Fonts link with graceful
fallback so the file also works fully offline.
"""
from __future__ import annotations

import datetime as _dt
import html
import logging
from pathlib import Path

from .dataset import EvalSample
from .perturb import PERTURBATION_TYPES
from .runner import ScorerReport

logger = logging.getLogger(__name__)

_TYPE_LABEL = {"entity_swap": "Entity swap", "number_perturb": "Number change",
               "negation_flip": "Negation flip",
               "foreign_sentence": "Foreign sentence"}

_CSS = """
:root{--paper:#FBFAF6;--card:#FFFFFF;--ink:#181F1D;--ink2:#5A625F;
--rule:#DCD9CE;--ok:#1D7A5F;--ok-bg:#E9F4EF;--bad:#B3372F;--bad-bg:#F9ECEA;
--mark:#F5D547;--mark-ink:#6B5A00;--mono:'IBM Plex Mono',ui-monospace,Consolas,monospace;
--serif:'Newsreader',Georgia,serif;--sans:'Public Sans',system-ui,sans-serif}
*{box-sizing:border-box;margin:0}
body{background:var(--paper);color:var(--ink);font:15px/1.65 var(--sans);
padding:0 20px 80px}
.wrap{max-width:1060px;margin:0 auto}
header{padding:48px 0 20px;border-bottom:3px double var(--ink)}
.masthead{display:flex;justify-content:space-between;align-items:flex-end;
flex-wrap:wrap;gap:12px}
h1{font:600 42px/1 var(--serif);letter-spacing:-.5px}
h1 .probe{color:var(--ok)}
.runmeta{font:12px/1.7 var(--mono);color:var(--ink2);text-align:right}
.tagline{font:italic 16px var(--serif);color:var(--ink2);margin-top:6px}
h2{font:600 22px var(--serif);margin:44px 0 6px}
.sub{color:var(--ink2);font-size:13.5px;margin-bottom:16px;max-width:70ch}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:10px;
padding:18px 20px 14px}
.card .name{font:600 12px var(--mono);text-transform:uppercase;
letter-spacing:.14em;color:var(--ink2)}
.card .f1{font:600 44px/1.15 var(--serif)}
.card .f1 small{font:400 14px var(--sans);color:var(--ink2)}
.kv{width:100%;border-collapse:collapse;margin-top:8px;font:12.5px var(--mono)}
.kv td{padding:3px 0;border-top:1px dashed var(--rule)}
.kv td:last-child{text-align:right}
table.matrix{border-collapse:collapse;width:100%;background:var(--card);
border:1px solid var(--rule);border-radius:10px;overflow:hidden;
font:13px var(--mono)}
.matrix th,.matrix td{padding:12px 14px;text-align:center;
border-bottom:1px solid var(--rule)}
.matrix th{background:var(--ink);color:var(--paper);font-weight:500;
letter-spacing:.05em}
.matrix td:first-child,.matrix th:first-child{text-align:left}
.matrix tr:last-child td{border-bottom:none}
.cell{display:inline-block;min-width:64px;padding:4px 8px;border-radius:6px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 22px}
.chip{font:12.5px var(--mono);border:1px solid var(--rule);background:var(--card);
padding:6px 13px;border-radius:999px;cursor:pointer;color:var(--ink2)}
.chip.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.case{background:var(--card);border:1px solid var(--rule);border-radius:10px;
padding:18px 22px;margin-bottom:14px}
.case-head{display:flex;gap:10px;align-items:center;flex-wrap:wrap;
margin-bottom:8px}
.badge{font:11px var(--mono);letter-spacing:.08em;text-transform:uppercase;
padding:3px 9px;border-radius:4px;border:1px solid}
.badge.clean{color:var(--ok);border-color:var(--ok);background:var(--ok-bg)}
.badge.contam{color:var(--bad);border-color:var(--bad);background:var(--bad-bg)}
.badge.type{color:var(--mark-ink);border-color:var(--mark);background:#FBF4D6}
.qid{font:11px var(--mono);color:var(--ink2);margin-left:auto}
.question{font:600 16px var(--serif);margin-bottom:10px}
.sent{padding:7px 10px 7px 14px;border-left:3px solid transparent;
border-radius:4px;margin:3px 0}
.sent.gt{border-left-color:var(--mark);background:#FDFBEF}
.sent .txt{display:inline}
.sent .marks{display:inline-flex;gap:6px;margin-left:10px;vertical-align:1px}
.stamp{font:10.5px var(--mono);padding:1px 7px;border-radius:3px}
.stamp.ok{color:var(--ok);background:var(--ok-bg)}
.stamp.flag{color:var(--bad);background:var(--bad-bg);
text-decoration:underline wavy var(--bad) 1px;text-underline-offset:3px}
.gt-tag{font:10.5px var(--mono);color:var(--mark-ink);margin-left:8px}
details{margin-top:10px}
summary{cursor:pointer;font:12.5px var(--mono);color:var(--ink2)}
.passage{font:13.5px/1.7 var(--serif);color:var(--ink2);background:var(--paper);
border:1px dashed var(--rule);border-radius:6px;padding:12px 16px;margin-top:8px}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--rule);
font:12.5px/1.8 var(--mono);color:var(--ink2)}
.hide{display:none}
@media(max-width:640px){h1{font-size:30px}.runmeta{text-align:left}}
"""

_JS = """
const chips=document.querySelectorAll('.chip');
chips.forEach(c=>c.addEventListener('click',()=>{
  chips.forEach(x=>x.classList.remove('on'));c.classList.add('on');
  const f=c.dataset.filter;
  document.querySelectorAll('.case').forEach(el=>{
    el.classList.toggle('hide', f!=='all' && !el.dataset.tags.includes(f));
  });
}));
"""


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _matrix_cell(v: float | None) -> str:
    if v is None:
        return "<td>—</td>"
    # ink-density scale: recall maps to background alpha of the verdict green
    alpha = 0.08 + 0.72 * v
    color = "#fff" if v > 0.55 else "var(--ink)"
    return (f'<td><span class="cell" style="background:rgba(29,122,95,{alpha:.2f});'
            f'color:{color}">{_pct(v)}</span></td>')


def _render_case(sample: EvalSample, per_scorer: dict[str, list[bool]],
                 max_missed_by: str) -> str:
    tags = ["contam" if sample.is_contaminated else "clean"]
    if sample.contamination_type:
        tags.append(sample.contamination_type)
    if max_missed_by:
        tags.append("disagree")
    badges = (f'<span class="badge contam">contaminated</span>'
              f'<span class="badge type">{_TYPE_LABEL.get(sample.contamination_type or "", "")}</span>'
              if sample.is_contaminated else '<span class="badge clean">clean</span>')
    rows = []
    for i, sent in enumerate(sample.answer_sentences):
        is_gt = sample.is_contaminated and i == sample.corrupted_sentence_idx
        marks = "".join(
            f'<span class="stamp {"flag" if flags[i] else "ok"}">'
            f'{name} {"✗" if flags[i] else "✓"}</span>'
            for name, flags in per_scorer.items())
        gt_tag = '<span class="gt-tag">◈ injected here</span>' if is_gt else ""
        rows.append(
            f'<div class="sent{" gt" if is_gt else ""}">'
            f'<span class="txt">{html.escape(sent)}</span>'
            f'<span class="marks">{marks}</span>{gt_tag}</div>')
    passage = html.escape(sample.passage)
    return (f'<div class="case" data-tags="{" ".join(tags)}">'
            f'<div class="case-head">{badges}'
            f'<span class="qid">{html.escape(sample.sample_id)}</span></div>'
            f'<div class="question">{html.escape(sample.question)}</div>'
            f'{"".join(rows)}'
            f'<details><summary>source passage</summary>'
            f'<div class="passage">{passage}</div></details></div>')


def render_report(samples: list[EvalSample], reports: list[ScorerReport],
                  out_path: Path, dataset_name: str = "SQuAD v2 (dev)",
                  seed: int | None = None, max_cases: int = 60) -> Path:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_clean = sum(1 for s in samples if not s.is_contaminated)
    n_cont = len(samples) - n_clean

    cards = "".join(
        f'<div class="card"><div class="name">{r.scorer} scorer</div>'
        f'<div class="f1">{r.f1:.2f}<small> F1</small></div>'
        f'<table class="kv">'
        f'<tr><td>precision</td><td>{r.precision:.3f}</td></tr>'
        f'<tr><td>recall</td><td>{r.recall:.3f}</td></tr>'
        f'<tr><td>false-alarm rate</td><td>{r.false_alarm_rate:.3f}</td></tr>'
        f'<tr><td>localization</td><td>{r.localization_accuracy:.3f}</td></tr>'
        f'<tr><td>runtime</td><td>{r.seconds}s</td></tr>'
        f'</table></div>' for r in reports)

    head = "".join(f"<th>{_TYPE_LABEL[t]}</th>" for t in PERTURBATION_TYPES)
    body = "".join(
        f'<tr><td>{r.scorer}</td>'
        + "".join(_matrix_cell(r.recall_by_type.get(t))
                  for t in PERTURBATION_TYPES)
        + f'{_matrix_cell(r.recall)}</tr>'
        for r in reports)

    # case files: prioritize disagreements and misses (the interesting ones)
    verdicts_by_scorer = {r.scorer: {a.sample_id: a for a in r.results}
                          for r in reports}

    def flags_for(s: EvalSample) -> dict[str, list[bool]]:
        return {name: [v.flagged for v in by_id[s.sample_id].verdicts]
                for name, by_id in verdicts_by_scorer.items()}

    def interest(s: EvalSample) -> int:
        answered = [by_id[s.sample_id].flagged
                    for by_id in verdicts_by_scorer.values()]
        disagree = len(set(answered)) > 1
        miss = s.is_contaminated and not all(answered)
        false_alarm = (not s.is_contaminated) and any(answered)
        return (3 if disagree else 0) + (2 if miss else 0) + (2 if false_alarm else 0)

    ordered = sorted(samples, key=interest, reverse=True)[:max_cases]
    cases = []
    for s in ordered:
        per = flags_for(s)
        answered = {n: any(f) for n, f in per.items()}
        disag = "disagree" if len(set(answered.values())) > 1 else ""
        cases.append(_render_case(s, per, disag))

    type_chips = "".join(
        f'<button class="chip" data-filter="{t}">{_TYPE_LABEL[t]}</button>'
        for t in PERTURBATION_TYPES)

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAGProbe — faithfulness audit</title>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,600;1,400&family=Public+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style></head><body><div class="wrap">
<header><div class="masthead"><div>
<h1>RAG<span class="probe">Probe</span></h1>
<div class="tagline">Is this answer actually in the sources? A meta-evaluated faithfulness audit.</div>
</div><div class="runmeta">run {now}<br>{dataset_name}
· seed {seed if seed is not None else "—"}<br>{n_clean} clean · {n_cont} contaminated answers</div>
</div></header>

<h2>Scorer scoreboard</h2>
<p class="sub">Answer-level detection of injected contamination. Localization
= among detected answers, how often the exact corrupted sentence was flagged.</p>
<div class="cards">{cards}</div>

<h2>Detection matrix</h2>
<p class="sub">Recall by contamination type — each scorer's fingerprint.
Darker ink = more of that failure mode caught. This is where cheap lexical
overlap and semantic entailment part ways.</p>
<table class="matrix"><tr><th>scorer</th>{head}<th>overall</th></tr>{body}</table>

<h2>Case files</h2>
<p class="sub">The {len(ordered)} most informative cases — disagreements,
misses, and false alarms first. ◈ marks the sentence we actually corrupted;
stamps show each scorer's sentence verdict.</p>
<div class="chips">
<button class="chip on" data-filter="all">all</button>
<button class="chip" data-filter="disagree">scorers disagree</button>
<button class="chip" data-filter="clean">clean</button>
<button class="chip" data-filter="contam">contaminated</button>
{type_chips}</div>
{"".join(cases)}

<footer>Methodology: grounded answers are constructed from source-passage
sentences (support guaranteed by construction); contaminated twins receive
exactly one seeded perturbation. Detection metrics are therefore measured
against provable ground truth. Limitation stated plainly: constructed
answers are near-verbatim, while real RAG answers paraphrase — treat these
numbers as an upper bound for extractive answers, not a universal
hallucination benchmark. Generated by RAGProbe.</footer>
</div><script>{_JS}</script></body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    logger.info("Report -> %s (%.0f KB)", out_path,
                out_path.stat().st_size / 1024)
    return out_path
