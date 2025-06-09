from __future__ import annotations

"""ideation.mix_and_match_strategy

End-to-end implementation of the **Mix-and-Match** ideation pipeline described
in `ideation/mix_and_match_design.md`.

Pipeline at a glance
--------------------
1. **Discovery** – Fetch a diverse set of trending papers from AlphaXiv.
2. **Comprehension (Stage 1)** – Use an LLM to produce a richly structured
   markdown summary for each paper (core innovation, methods, limitations, …).
   Results are cached on disk to avoid redundant API calls.
3. **Connection (Stage 2)** – For every cross-domain pair, ask the LLM to
   synthesize conceptual links, complementarities and risks using a
   five-step rubric (concept mapping → complementarity matrix → synergy
   hypotheses → novelty/feasibility JSON → risk factors).
4. **Ideation (Stage 3)** – The model drafts **early-stage Research Notes**
   (concise abstract, central question, approach, open questions) — the sort of
   pitch a scientist might email to peers.
5. **Persistence** – Everything is written to version-controlled markdown files
   under `ideation/ideas/` for further curation.

Design principles
-----------------
• **Provider-agnostic** – Works with OpenAI, Anthropic (Claude) or any
  OpenAI-compatible router (e.g. OpenRouter). Provider is inferred from the
  model string.
• **Small surface area** – A single `MixAndMatchEngine` manages orchestration; 
  logic is split into explicit, testable helper methods.
• **Deterministic & Incremental** – On-disk caches include a `default=str`
  JSON dump so datetime objects serialise cleanly; changing prompt templates
  invalidates caches via simple file versioning.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Sequence
import json
import logging
import random
from datetime import datetime

from utils.alphaxiv import get_trending_papers, AlphaPaper
from utils import arxiv as arxiv_api
from utils.llm import UnifiedLLM, Message, ModelType, LLMProvider

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Data-model helpers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AnalyzedPaper:
    """Container holding metadata + Stage-1 analysis."""

    alpha: AlphaPaper
    arxiv: arxiv_api.ArxivPaper | None
    analysis: str  # raw Stage-1 LLM output

    @property
    def short_id(self) -> str:  # e.g. "2406.01234"
        return self.alpha.arxiv_id or self.alpha.alpha_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": asdict(self.alpha),
            "arxiv": asdict(self.arxiv) if self.arxiv else None,
            "analysis": self.analysis,
        }

# ---------------------------------------------------------------------------
# Prompt templates (kept simple, could be moved to a dedicated module)
# ---------------------------------------------------------------------------

_STAGE1_PROMPT = (
    "You are an expert interdisciplinary research analyst. Your goal is to extract *actionable* knowledge from the paper below for downstream ideation.\n\n"
    "Return a **concise markdown document** (< 250 words) that strictly uses the following section headings *in this order*:\n\n"
    "### Core Innovation\n"
    "### Problem Addressed\n"
    "### Methodological Highlights\n"
    "### Key Findings\n"
    "### Limitations & Open Questions\n"
    "### Transferable Techniques (≥3 bullet points)\n\n"
    "If a finding is surprising or counter-intuitive, prefix the bullet with the emoji ⚡.\n\n"
    "---\n\n"
    "**Paper metadata**\n"
    "Title: {title}\n"
    "Category: {category}\n"
    "Published: {published}\n\n"
    "**Abstract**\n"
    "{abstract}\n"
)

_STAGE2_PROMPT = (
    "You are a seasoned cross-disciplinary scientist tasked with synthesising insights from two papers. Follow the **five-step framework** below and answer in markdown using the exact headings shown.\n\n"
    "### 1. Conceptual Mapping\n"
    "Provide a bullet list mapping key concepts/techniques from Paper A to analogous concepts in Paper B.\n\n"
    "### 2. Complementarity Matrix\n"
    "Create a 2 × 2 table (Markdown) where rows = strengths of Paper A, columns = strengths of Paper B. Fill each cell with how the row can compensate the column's weakness.\n\n"
    "### 3. Synergy Hypotheses (≥3)\n"
    "Each hypothesis should be a single sentence describing how combining the papers could unlock new research avenues.\n\n"
    "### 4. Novelty & Feasibility Scores\n"
    "Give a JSON object with keys `novelty` and `feasibility`, each an integer 0-10 (higher is better).\n\n"
    "### 5. Risk Factors\n"
    "List the top technical or conceptual risks in pursuing these synergies.\n\n"
    "---\n\n"
    "[Paper A Analysis]\n{analysis_a}\n\n"
    "[Paper B Analysis]\n{analysis_b}\n"
)

_STAGE3_PROMPT = (
    "Using the synthesis above, propose **exactly five** *distinct* research ideas in the style of an early-stage *Research Note* that you might send to colleagues for feedback.\n\n"
    "For **each idea** output the following markdown structure **verbatim** (replace bracketed sections). Keep the entire block for an idea ≤ 200 words.\n\n"
    "---\n"
    "#### Idea {n}: <Compelling Title>\n\n"
    "**Research Abstract (≤ 60 words)**\n"
    "<concise abstract here>\n\n"
    "**Date:** <YYYY-MM-DD>\n"
    "**Papers Inspiring This:** <PaperA arXiv ID> & <PaperB arXiv ID>\n"
    "**The Question**\n"
    "<1-2 sentence core research question>\n\n"
    "**Why It's Interesting**\n"
    "<bullet list of 2-3 reasons this matters>\n\n"
    "**Sketch of Approach**\n"
    "<2-3 bullet steps outlining experimental plan / methodology>\n\n"
    "**Resources Needed**\n"
    "<datasets, compute, collaborators>\n\n"
    "**Open Questions**\n"
    "<bullet list of uncertainties / risks>\n"
    "---\n"
    "Repeat the template for Idea 1 – 5, replacing placeholders. Output nothing else.\n"
)


# ---------------------------------------------------------------------------
# Engine implementation
# ---------------------------------------------------------------------------

class MixAndMatchEngine:
    """End-to-end pipeline orchestrator."""

    def __init__(
        self,
        llm: UnifiedLLM | None = None,
        *,
        idea_dir: str | Path = "ideation/ideas",
        cache_dir: str | Path = "ideation/cache",
        openai_model: ModelType | str | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.llm = llm or self._default_llm()
        self.idea_dir = Path(idea_dir)
        self.cache_dir = Path(cache_dir)
        self.idea_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if random_seed is not None:
            random.seed(random_seed)
        # Timestamp for this run – used to avoid filename clashes
        self._run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Normalise model argument – can be str or ModelType
        if openai_model is None:
            self._model: ModelType = ModelType.GPT_4O
        elif isinstance(openai_model, ModelType):
            self._model = openai_model
        else:
            self._model = ModelType(openai_model)

        # Determine provider from model if not explicitly given
        self._provider: LLMProvider = self._infer_provider(self._model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        num_trending: int = 30,
        target_domains: int = 5,
        papers_per_domain: int = 4,
        max_pairs: int = 20,
    ) -> None:
        """Run the full pipeline and persist ideas to *idea_dir*."""
        papers = self._fetch_trending(num_trending)
        selected = self._select_diverse(papers, target_domains, papers_per_domain)
        analyzed = self._analyze_papers(selected)
        pairs = self._pair_papers(analyzed, max_pairs)
        for idx, (p1, p2) in enumerate(pairs, start=1):
            try:
                self._ideate_pair(idx, p1, p2)
            except Exception as exc:  # noqa: BLE001 – continue processing other pairs
                logger.error("Ideation failed for pair %s / %s: %s", p1.short_id, p2.short_id, exc)

    # ------------------------------------------------------------------
    # Step 1 – Fetch & select papers
    # ------------------------------------------------------------------

    def _fetch_trending(self, num_papers: int) -> List[AlphaPaper]:
        logger.info("Fetching %d trending papers from AlphaXiv…", num_papers)
        batch_size = min(num_papers, 50)
        all_papers: List[AlphaPaper] = []
        page = 0
        while len(all_papers) < num_papers:
            remaining = num_papers - len(all_papers)
            page_size = min(batch_size, remaining)
            batch = get_trending_papers(page_num=page, page_size=page_size, sort_by="Hot")
            if not batch:
                break  # API exhausted
            all_papers.extend(batch)
            page += 1
        logger.info("Retrieved %d papers", len(all_papers))
        return all_papers[:num_papers]

    def _select_diverse(
        self,
        papers: Sequence[AlphaPaper],
        target_domains: int,
        per_domain: int,
    ) -> List[AlphaPaper]:
        """Select a diverse subset across *target_domains* categories."""
        logger.info("Selecting a diverse subset (%d domains × %d)…", target_domains, per_domain)
        # Group by first category for simplicity (could be improved)
        domain_map: Dict[str, List[AlphaPaper]] = {}
        for p in papers:
            domain = (p.custom_categories or p.categories or ["misc"])[0]
            domain_map.setdefault(domain, []).append(p)

        # Sort domains by popularity to ensure enough papers
        domains_sorted = sorted(domain_map.items(), key=lambda kv: len(kv[1]), reverse=True)
        chosen_domains = [d for d, _ in domains_sorted[:target_domains]]

        selected: List[AlphaPaper] = []
        for domain in chosen_domains:
            pool = domain_map[domain]
            random.shuffle(pool)
            selected.extend(pool[:per_domain])

        logger.info("Selected %d papers across %d domains", len(selected), len(chosen_domains))
        return selected

    # ------------------------------------------------------------------
    # Step 2 – Deep content analysis
    # ------------------------------------------------------------------

    def _analyze_papers(self, papers: Sequence[AlphaPaper]) -> List[AnalyzedPaper]:
        results: List[AnalyzedPaper] = []
        for paper in papers:
            cache_path = self.cache_dir / f"{paper.arxiv_id or paper.alpha_id}.json"
            if cache_path.exists():
                data = json.loads(cache_path.read_text())
                analysis_text = data["analysis"]
                arxiv_paper = (
                    arxiv_api.ArxivPaper(**data["arxiv"]) if data.get("arxiv") else None
                )
                logger.debug("Loaded cached analysis for %s", paper.arxiv_id)
            else:
                arxiv_paper = arxiv_api.get_paper(paper.arxiv_id)
                analysis_text = self._run_stage1(arxiv_paper or paper)
                cache_path.write_text(
                    json.dumps(
                        {
                            "analysis": analysis_text,
                            "arxiv": asdict(arxiv_paper) if arxiv_paper else None,
                        },
                        indent=2,
                        default=str,
                    )
                )
            results.append(AnalyzedPaper(alpha=paper, arxiv=arxiv_paper, analysis=analysis_text))
        return results

    def _run_stage1(self, paper: AlphaPaper | arxiv_api.ArxivPaper) -> str:
        """LLM call for Stage 1 (paper comprehension)."""
        title = paper.title
        abstract = getattr(paper, "summary", None) or getattr(paper, "abstract", None) or ""
        category = getattr(paper, "categories", ["Unknown"])[0] if hasattr(paper, "categories") else "Unknown"
        published = getattr(paper, "published", "Unknown")
        prompt = _STAGE1_PROMPT.format(
            title=title, 
            abstract=abstract,
            category=category,
            published=published
        )
        messages = [Message(role="user", content=prompt)]
        resp = self.llm.complete(messages, model=self._model, provider=self._provider)
        return resp.content.strip()

    # ------------------------------------------------------------------
    # Step 3 – Intelligent pairing
    # ------------------------------------------------------------------

    def _pair_papers(
        self,
        analyzed: Sequence[AnalyzedPaper],
        max_pairs: int,
    ) -> List[Tuple[AnalyzedPaper, AnalyzedPaper]]:
        """Create cross-domain pairs, avoiding intra-domain matches when possible."""
        pairs: List[Tuple[AnalyzedPaper, AnalyzedPaper]] = []
        for i, p1 in enumerate(analyzed):
            for p2 in analyzed[i + 1 :]:
                # Skip if same primary category (rough heuristic)
                cat1 = (p1.alpha.custom_categories or p1.alpha.categories or [None])[0]
                cat2 = (p2.alpha.custom_categories or p2.alpha.categories or [None])[0]
                if cat1 == cat2:
                    continue
                pairs.append((p1, p2))
        random.shuffle(pairs)
        return pairs[:max_pairs]

    # ------------------------------------------------------------------
    # Step 4 & 5 – Ideation + persistence
    # ------------------------------------------------------------------

    def _build_stage1_prompt(self, paper: AlphaPaper | arxiv_api.ArxivPaper) -> str:
        """Return the user prompt used for Stage 1 given *paper*."""
        title = paper.title
        abstract = getattr(paper, "summary", None) or getattr(paper, "abstract", None) or ""
        category = getattr(paper, "categories", ["Unknown"])[0] if hasattr(paper, "categories") else "Unknown"
        published = getattr(paper, "published", "Unknown")
        return _STAGE1_PROMPT.format(
            title=title,
            abstract=abstract,
            category=category,
            published=published
        )

    def _ideate_pair(
        self,
        idx: int,
        p1: AnalyzedPaper,
        p2: AnalyzedPaper,
    ) -> None:
        logger.info("\n[%d] Ideating for pair %s ↔ %s", idx, p1.short_id, p2.short_id)

        # ------------------------------------------------------------------
        # Build conversation with Stage-1 history for *both* papers
        # ------------------------------------------------------------------
        convo: List[Message] = []

        # Paper A – Stage 1 messages
        stage1_prompt_a = self._build_stage1_prompt(p1.arxiv or p1.alpha)
        convo.append(Message(role="user", content=stage1_prompt_a))
        convo.append(Message(role="assistant", content=p1.analysis))

        # Paper B – Stage 1 messages
        stage1_prompt_b = self._build_stage1_prompt(p2.arxiv or p2.alpha)
        convo.append(Message(role="user", content=stage1_prompt_b))
        convo.append(Message(role="assistant", content=p2.analysis))

        # Stage 2 – connections
        stage2_prompt = _STAGE2_PROMPT.format(analysis_a=p1.analysis, analysis_b=p2.analysis)
        convo.append(Message(role="user", content=stage2_prompt))
        stage2_resp = self.llm.complete(convo, model=self._model, provider=self._provider)
        convo.append(Message(role="assistant", content=stage2_resp.content))

        # Stage 3 – ideation
        convo.append(Message(role="user", content=_STAGE3_PROMPT))
        stage3_resp = self.llm.complete(convo, model=self._model, provider=self._provider)

        # Persist to Markdown
        out_path = (
            self.idea_dir
            / f"{self._run_ts}_pair{idx:02d}_{p1.short_id}_{p2.short_id}.md"
        ).resolve()
        md = self._build_markdown(p1, p2, stage2_resp.content, stage3_resp.content)
        out_path.write_text(md)
        try:
            display_path = out_path.relative_to(Path.cwd())
        except ValueError:
            display_path = out_path
        logger.info("Saved ideas → %s", display_path)

        # --------------------------------------------------------------
        # Append ideas to CSV for easy downstream querying
        # --------------------------------------------------------------
        ideas_csv = (self.idea_dir / "ideas.csv").resolve()
        ideas = self._parse_ideas(stage3_resp.content)
        from csv import DictWriter
        fieldnames = [
            "title",
            "abstract",
            "central_question",
            "approach_summary",
            "resources_needed",
            "paper_a_id",
            "paper_b_id",
            "date_generated",
        ]

        with ideas_csv.open("a", newline="") as fh:
            writer = DictWriter(fh, fieldnames=fieldnames)
            if fh.tell() == 0:
                writer.writeheader()
            for idea in ideas:
                idea.update({
                    "paper_a_id": p1.short_id,
                    "paper_b_id": p2.short_id,
                })
                writer.writerow({k: idea.get(k, "") for k in fieldnames})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_llm() -> UnifiedLLM:
        """Instantiate a UnifiedLLM with all providers that have API keys set."""
        llm = UnifiedLLM()
        # Always attempt OpenAI – raises if key missing
        try:
            llm.add_provider(LLMProvider.OPENAI, default=True)
        except ValueError:
            pass

        # Add Anthropic if available
        try:
            llm.add_provider(LLMProvider.ANTHROPIC)
        except ValueError:
            pass

        # Add OpenRouter if available
        try:
            llm.add_provider(LLMProvider.OPENROUTER)
        except ValueError:
            pass

        if not llm.providers:
            raise RuntimeError(
                "No LLM providers configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY."
            )

        return llm

    # ------------------------------------------------------------------
    # Provider inference helper
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_provider(model: ModelType) -> LLMProvider:
        """Heuristically map a ``ModelType`` to its corresponding provider."""
        name = model.value  # raw string identifier
        if name.startswith("anthropic/") or name.startswith("google/") or "claude" in name.lower():
            # If model routed via OpenRouter (anthropic/xxx) treat as OpenRouter
            if name.startswith("anthropic/") or name.startswith("openai/") or "/" in name:
                return LLMProvider.OPENROUTER
            return LLMProvider.ANTHROPIC
        if name.startswith("openai/") or name.startswith("qwen/") or name.startswith("google/"):
            return LLMProvider.OPENROUTER
        # Default to OpenAI for GPT/O models
        return LLMProvider.OPENAI

    @staticmethod
    def _build_markdown(
        p1: AnalyzedPaper,
        p2: AnalyzedPaper,
        connections: str,
        ideas: str,
    ) -> str:
        def _fmt_paper(p: AnalyzedPaper) -> str:
            url = p.alpha.url or p.arxiv.html_url if p.arxiv else ""
            return f"### {p.alpha.title}\n\n- arXiv ID: `{p.alpha.arxiv_id}`\n- URL: {url}\n\n#### LLM Analysis\n{p.analysis}\n"

        md_parts = [
            f"# Mix-and-Match Ideation — {p1.short_id} × {p2.short_id}\n",
            "## Papers\n",
            _fmt_paper(p1),
            _fmt_paper(p2),
            "## Connections (Stage 2)\n",
            connections.strip() + "\n",
            "## Generated Ideas (Stage 3)\n",
            ideas.strip() + "\n",
        ]
        return "\n".join(md_parts)

    # ------------------------------------------------------------------
    # Idea parsing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ideas(markdown_block: str) -> List[Dict[str, str]]:
        """Extract ideas from Stage-3 markdown into structured dicts."""
        import re, datetime

        ideas: List[Dict[str, str]] = []
        blocks = markdown_block.split("\n---")
        for block in blocks:
            block = block.strip()
            if not block.startswith("#### Idea"):
                continue

            title_match = re.search(r"#### Idea \d+: (.+)", block)
            title = title_match.group(1).strip() if title_match else ""

            def extract_section(name: str) -> str:
                pattern = rf"\*\*{name}\*\*\s*\n(.+?)(?:\n\s*\*\*|$)"
                m = re.search(pattern, block, re.DOTALL)
                return m.group(1).strip() if m else ""

            abstract = extract_section(r"Research Abstract \(≤ 60 words\)")
            central_q = extract_section("The Question")
            approach = extract_section("Sketch of Approach")
            resources = extract_section("Resources Needed")
            date_field = extract_section("Date:") or datetime.date.today().isoformat()

            ideas.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "central_question": central_q,
                    "approach_summary": approach,
                    "resources_needed": resources,
                    "date_generated": date_field,
                }
            )
        return ideas


# ---------------------------------------------------------------------------
# CLI entry-point (python -m ideation.mix_and_match_strategy)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run the mix-and-match ideation pipeline.")
    parser.add_argument("--num", type=int, default=30, help="Number of trending papers to fetch")
    parser.add_argument("--domains", type=int, default=5, help="Target number of distinct domains")
    parser.add_argument("--per-domain", type=int, default=4, help="Papers per domain")
    parser.add_argument("--pairs", type=int, default=20, help="Maximum paper pairs to ideate on")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model identifier to use (see utils.llm.ModelType)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    engine = MixAndMatchEngine(openai_model=args.model, random_seed=args.seed)
    engine.run(
        num_trending=args.num,
        target_domains=args.domains,
        papers_per_domain=args.per_domain,
        max_pairs=args.pairs,
    )
