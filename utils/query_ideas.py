from __future__ import annotations

"""ideation.query_ideas

Tiny utility to load the aggregated `ideas.csv` and run simple keyword queries
across title, abstract and central question.

Usage (module mode)
-------------------
```bash
python -m utils.query_ideas --query "sparse attention" --limit 10
```
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]  # project root
IDEAS_CSV = ROOT / "ideation" / "ideas" / "ideas.csv"

# Optional fancy CLI rendering via rich -------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    _RICH_AVAILABLE = True
    _console = Console()
except ImportError:  # pragma: no cover – rich is optional
    _RICH_AVAILABLE = False


def load_ideas() -> List[Dict[str, str]]:
    if not IDEAS_CSV.exists():
        raise FileNotFoundError("ideas.csv not found – run the mix_and_match pipeline first.")
    with IDEAS_CSV.open() as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def search_ideas(rows: List[Dict[str, str]], pattern: str) -> List[Dict[str, str]]:
    regex = re.compile(pattern, re.IGNORECASE)
    matches = []
    for r in rows:
        haystack = " ".join([r.get("title", ""), r.get("abstract", ""), r.get("central_question", "")])
        if regex.search(haystack):
            matches.append(r)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Query generated research ideas.")
    parser.add_argument("--query", help="Regex / keyword to search for (omit for interactive mode)")
    parser.add_argument("--limit", type=int, default=20, help="Max rows to display")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive search shell")
    args = parser.parse_args()

    rows = load_ideas()

    def pretty_print(matches):
        if not matches:
            print("No ideas matched your query.")
            return
        if _RICH_AVAILABLE:
            tbl = Table(show_header=True, header_style="bold magenta")
            tbl.add_column("#", style="dim", width=3)
            tbl.add_column("Title", overflow="fold")
            tbl.add_column("Question", overflow="fold")
            tbl.add_column("Approach", overflow="fold")
            for i, r in enumerate(matches, 1):
                tbl.add_row(
                    str(i),
                    r["title"],
                    r["central_question"][:80] + ("…" if len(r["central_question"]) > 80 else ""),
                    r["approach_summary"][:80] + ("…" if len(r["approach_summary"]) > 80 else ""),
                )
            _console.print(tbl)
        else:
            for i, r in enumerate(matches, 1):
                print(f"{i}. {r['title']}  (from {r['paper_a_id']} × {r['paper_b_id']})")
                print(f"   Question: {r['central_question'][:120]}…")
                print(f"   Approach: {r['approach_summary'][:120]}…\n")

    if args.interactive or not args.query:
        print("Interactive idea search – type a keyword/regex, or 'quit' to exit\n")
        try:
            while True:
                user_input = input("search> ").strip()
                if user_input.lower() in {"q", "quit", "exit"}:
                    break
                pretty_print(search_ideas(rows, user_input)[: args.limit])
        except KeyboardInterrupt:
            print()
            return
    else:
        pretty_print(search_ideas(rows, args.query)[: args.limit])


if __name__ == "__main__":
    main()
