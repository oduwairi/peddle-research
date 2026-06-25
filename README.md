# Peddle — Domain-Specialized Agentic LLM for Marketing Campaign Generation

Research code accompanying the M.Sc. thesis *"A Domain-Specialized Agentic LLM for
Marketing Campaign Generation"* (O. ALDuwairi). This repository contains the full
data-construction, scoring, fine-tuning, and evaluation pipeline used to produce the
results reported in the thesis.

> This is a **research release**: it excludes the production web application (the
> deployed Next.js product) and ships a de-identified data subset. See
> [DATA.md](DATA.md) for the data-availability statement.

## Research questions

- **RQ1** — Can a fine-tuned open-source 8B model (Qwen3-8B + QLoRA) match or surpass
  a frontier model at generating ad copy?
- **RQ2** — What is the cumulative vs. independent effect of fine-tuning and a live
  RAG/agent workflow on output quality?
- **RQ3** — Can proxy signals from campaign metadata (longevity, engagement velocity,
  early stoppage) reliably predict real campaign performance?

## Pipeline → code map

The system is built in three phases. Each maps to packages under `src/draper/`:

| Phase | Stage | Package | Driver script | Config |
|-------|-------|---------|---------------|--------|
| 1 — Corpus | Scrape ad libraries | `scraping/`, `collection/` | `scripts/scrape.py`, `scripts/collect.py` | — |
| 1 — Corpus | Proxy scoring + tiering (KM survival + engagement) | `scoring/` | `scripts/score.py` | `configs/scoring.yaml` |
| 1 — Corpus | Instruction-backtranslation → SFT pairs | `construction/` | `scripts/construct.py` | `configs/construction.yaml` |
| 2 — Training | QLoRA fine-tune (Qwen3-8B) | `training/` | `scripts/train.py` | `configs/training.yaml` |
| 3 — Eval | Learned scorer (text-only proxy predictor) | `scoring_predictor/` | `scripts/predict.py`, `serve_scoring_predictor.py` | — |
| 3 — Eval | 2×2 ablation, MAUVE, learned-scorer, validation sets | `evaluation/` | `scripts/eval.py` | `configs/eval.yaml` |

The agent/RAG arm (research tools `scrape_url`, `web_search`, `exa_similar`; writing
tools `draft_campaign`, `ask_draper`, `generate_image`; scoring `score_copy`; output
`emit_campaign`) was implemented in the production frontend and is **not** included in
this research release; its design is documented in
[`docs/project/AGENT_ARCHITECTURE.md`](docs/project/AGENT_ARCHITECTURE.md).

## Layout

```
src/draper/          # research code (scraping → scoring → construction → training → eval)
scripts/             # CLI drivers for each pipeline stage
configs/             # YAML configs (scoring, construction, training, eval)
tests/               # unit + contract tests per package
docs/project/        # architecture & methodology notes
docs/training/       # training-run postmortems
docs/api/ADFLEX.md   # AdFlex API integration notes
```

## Setup

```bash
# Python 3.11+, uv (https://docs.astral.sh/uv/)
uv sync                       # install from pyproject.toml / uv.lock
cp .env.example .env          # fill in provider keys (see below)
```

All credentials are read from environment variables — none are committed. Required for
a full run: AdFlex API key (scraping), an OpenAI-compatible teacher endpoint
(construction), and a GPU host for QLoRA training (see `scripts/train_cloud.sh`).

## Data

The full 55,000-ad AdFlex corpus is **not redistributed** (third-party paid-API terms).
This release ships the derived SFT training set and a de-identified sample so the
training/eval steps can be inspected and re-run. See **[DATA.md](DATA.md)**.

## Datasets used for validation (public)

- Upworthy Research Archive — Matias et al. 2021, *Nature Scientific Data* (DOI 10.17605/OSF.IO/JD64P)
- Internet Research Agency (IRA) Facebook Ads — https://github.com/umd-mith/irads
- Meta Ad Library, Google Ads Transparency Center, TikTok Ad Library, BigSpy (cross-checking)

## Citation

```bibtex
@mastersthesis{alduwairi2026peddle,
  title  = {A Domain-Specialized Agentic LLM for Marketing Campaign Generation},
  author = {ALDuwairi, Osama},
  year   = {2026}
}
```
