"""RAGProbe CLI.

  ragprobe build --n 200 --seed 7 --out data/eval.jsonl
  ragprobe run --dataset data/eval.jsonl --scorers lexical,nli \\
               --report reports/report.html --json reports/meta.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .dataset import build_clean_samples, download_squad, read_jsonl, write_jsonl
from .perturb import contaminate
from .report import render_report
from .runner import run_scorer
from .scorers import get_scorer


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")


def cmd_build(args: argparse.Namespace) -> None:
    squad = download_squad(Path(args.squad_cache))
    clean = build_clean_samples(squad, n=args.n, seed=args.seed)
    contaminated = contaminate(clean, seed=args.seed)
    write_jsonl(clean + contaminated, Path(args.out))


def cmd_run(args: argparse.Namespace) -> None:
    samples = read_jsonl(Path(args.dataset))
    seed = args.seed
    reports = []
    for name in args.scorers.split(","):
        scorer = get_scorer(name.strip())
        reports.append(run_scorer(scorer, samples))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            [r.summary_dict() for r in reports], indent=2))
        logging.getLogger(__name__).info("Meta-eval JSON -> %s", args.json)
    render_report(samples, reports, Path(args.report), seed=seed)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(prog="ragprobe",
                                description="Faithfulness evaluation for RAG answers")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a contamination eval set")
    b.add_argument("--n", type=int, default=200, help="clean samples (contaminated twins added on top)")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--out", default="data/eval.jsonl")
    b.add_argument("--squad-cache", default="data/squad_dev_v2.json")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run", help="run scorers + meta-evaluation + report")
    r.add_argument("--dataset", default="data/eval.jsonl")
    r.add_argument("--scorers", default="lexical,nli",
                   help="comma list: lexical,nli,judge")
    r.add_argument("--report", default="reports/report.html")
    r.add_argument("--json", default="reports/meta.json")
    r.add_argument("--seed", type=int, default=None, help="displayed in report")
    r.set_defaults(fn=cmd_run)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
