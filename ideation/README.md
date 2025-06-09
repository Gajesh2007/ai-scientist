# Ideation Module

The **Ideation** layer of *Gaj's AI Lab* is responsible for generating, curating
and ranking novel research ideas by leveraging state-of-the-art language models
plus up-to-date literature from AlphaXiv, arXiv and Semantic Scholar.

Currently the module implements a single end-to-end pipeline:

### ➡️ `mix_and_match_strategy.py`
A provider-agnostic pipeline that automatically:
1. Fetches trending papers from **AlphaXiv**
2. Pulls full metadata / abstracts from **arXiv**
3. Summarises each paper via an LLM (Stage 1)
4. Finds cross-domain paper pairs and synthesises connections (Stage 2)
5. Drafts early-stage *Research Notes* for each pair (Stage 3)
6. Stores everything in markdown under `ideation/ideas/`

### Running the pipeline
```bash
python -m ideation.mix_and_match_strategy \
  --num 20            # number of trending papers
  --domains 4         # how many distinct domains to cover
  --per-domain 3       # papers per domain
  --pairs 15          # max pairs to ideate on
  --model gpt-4o      # any model present in utils.llm.ModelType
```
Environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) are
automatically detected.

### Output structure
```
ideation/
 └─ ideas/
     ├─ idea_pair_01_<idA>_<idB>.md
     ├─ idea_pair_02_<idA>_<idB>.md
     └─ …
```
Each markdown file contains:
* Raw Stage 1 analyses for both papers
* Structured Stage 2 synthesis (concept mapping, synergy hypotheses, …)
* Five peer-review-ready *Research Notes* produced in Stage 3

### Roadmap
* Ranking / triage step based on novelty × feasibility JSON
* Embedding-based diversity selection and idea clustering
* Async execution for faster throughput

---
Contributions and issue reports are welcome. 🎉
