# code4ed-resume-analyser

> Scores a resume against a job description and names the specific skills that are
> missing, for placement teams triaging more applications than they can read.

**36th of 1037 teams — Code4Edtech Challenge, Innomatics Research Labs.**

![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)

![Resume-JD Analyzer](docs/screenshots/analysis.png)

## The problem

A placement cell receives a few thousand resumes against a few dozen roles. Nobody
reads them all, so the triage happens by keyword search — and keyword search is
exactly the wrong tool for this.

Resumes and job descriptions describe the same work in different vocabularies. A JD
asks for "containerisation"; the resume says "Docker". The JD wants "RDBMS
experience"; the resume lists "PostgreSQL". Exact matching misses both, and the
candidate is filtered out for a skill they have. Push the other way and use pure
embedding similarity, and the opposite failure appears: everything looks vaguely
related to everything, a resume full of adjacent-sounding language scores well, and
the ranking stops discriminating.

There is a second problem underneath the first. A score on its own is not
actionable. A student told they scored 62 learns nothing they can act on. A student
told they are missing Docker, Kubernetes and CI/CD experience for this specific
role has a list they can work through. The output has to be a gap, not a grade.

## Approach

Scoring runs on two paths, and which one you get depends on whether a language
model is configured.

**The deterministic path, which always runs.** Exact keyword overlap, TF-IDF cosine
similarity and fuzzy string matching (`keyword_matcher.py`) establish what the
resume literally says. On top of that `scorer.py` applies a hand-weighted rubric:
skill categories carry fixed weights — programming 0.25, frameworks 0.20, databases
and cloud 0.15 each, then smaller terms — combined with separate experience,
education, keyword-density and domain-relevance components. It is a rubric, not a
learned model, and it is explainable line by line.

**The language-model path, when `GOOGLE_API_KEY` is set.** Gemini evaluates the
resume against the JD and returns an overall score plus narrative feedback, which
`process_llm_evaluation` reconciles against the deterministically extracted
experience and education rather than trusting outright.

**The output is a skill gap, not just a number.** The matcher tracks which JD
requirements were satisfied and by what, so an unmatched requirement is reported as
a named missing skill rather than disappearing into an aggregate. That is the part
a student can act on.

### What is not wired up

`vector_store.py` exposes a `VectorStore` whose ChromaDB backend is disabled — the
file says so, citing SQLite version conflicts on Streamlit Cloud. Its `client` is
always `None`, `calculate_resume_jd_similarity` returns `0.0`, and every call site
in `langchain_pipeline.py` and `langgraph_workflow.py` is guarded by
`if vector_store.client:` and therefore never executes. There is no embedding
similarity in the scoring path today, so vocabulary mismatch is handled by fuzzy
matching and the LLM, not by vectors. `KEYWORD_WEIGHT` and `SEMANTIC_WEIGHT` are
read into `Config` and used nowhere.

### Degrading without a language model

The LLM layer is optional by design. When `GOOGLE_API_KEY` is absent the app reports
`LLM Service: Unavailable / Hybrid Fallback: Active` in the sidebar and continues on
the deterministic path — parsing, keyword matching, TF-IDF, fuzzy matching and
vector similarity all run locally. The language model adds narrative feedback; it is
not load-bearing for the score. A placement cell without an API budget still gets a
working triage tool, which is the point.

## Features

| Capability | How it works |
|---|---|
| PDF and DOCX parsing | `pdf_parser` and `docx_parser` with a `docx2txt` fallback when python-docx yields nothing |
| Keyword matching | Exact overlap, TF-IDF cosine similarity and fuzzy matching via `fuzzywuzzy` |
| Weighted rubric | Fixed per-category skill weights plus experience, education, keyword-density and domain-relevance components |
| Optional LLM evaluation | Gemini scores and explains; its extracted experience and education are reconciled against the deterministic parse |
| Skill-gap output | Unmatched JD requirements surfaced as named missing skills |
| Live extraction preview | Skills detected and displayed as the file is parsed, before scoring completes |
| Graceful LLM fallback | Full deterministic pipeline when no API key is configured |
| Batch processing | Multiple resumes scored against one saved job description |

## Screenshots

### Job description

![Job description](docs/screenshots/job-description.png)

Step 1 accepts a pasted JD, or loads the bundled sample. The JD is saved before any
resume is uploaded, so one description can be reused across a batch.

### Live extraction

![Live extraction](docs/screenshots/analysis.png)

Step 2 with a synthetic test resume uploaded. Skills are detected and shown as the
document is parsed — 15 found here, including `fastapi`, `azure`, `postgresql` and
`tensorflow`. The sidebar shows the LLM service unavailable and the hybrid fallback
active; extraction is unaffected.

### Entry point

![Landing](docs/screenshots/hero.png)

The three-step flow is explicit rather than a single upload box, because the JD is
the reusable half of the pair.

## Architecture

```
resume (PDF / DOCX)            job description (text)
        │                              │
        ▼                              ▼
   file_parser                   text_processor
   pdf_parser / docx_parser      requirement extraction
        │                              │
        └──────────────┬───────────────┘
                       ▼
              ┌──────────────────────────┐
              │  keyword_matcher         │  exact · TF-IDF · fuzzy
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │  scorer (weighted rubric)│  skills by category · experience
              │                          │  education · keyword density
              └────────────┬─────────────┘  domain relevance
                           │
                           ├──► relevance score
                           └──► named missing skills

              llm_service ─ optional: Gemini score + narrative,
                            reconciled against the deterministic parse.
                            Skipped entirely when no key is set.

              vector_store ─ present but disabled; client is always None
                             and no call site executes.
```

Parsing, matching and scoring are separate services under `app/services/`, so the
LLM layer can be removed from the path without touching the scoring logic. That
separation is what makes the keyless fallback a configuration state rather than a
degraded code path.

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| UI | Streamlit | A placement cell needs a URL, not a deployment; Streamlit gets there in one file |
| Parsing | pdfplumber, python-docx, docx2txt | Two independent DOCX paths because real resumes are inconsistently authored |
| Matching | scikit-learn, fuzzywuzzy | TF-IDF, cosine similarity and fuzzy ratios — cheap, and explainable to a student |
| LLM | Google Gemini via LangChain | Narrative feedback only; the score never depends on it |
| Storage | SQLite | Saved job descriptions and evaluation history; no server to run |

## Getting started

### Prerequisites

- Python 3.11
- No API key required — the app runs fully on its deterministic path without one

### Installation

```bash
git clone https://github.com/varadharajanv0310/code4ed-resume-analyser.git
cd code4ed-resume-analyser

python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt               # macOS / Linux
```

### Configuration

Optional. Copy `.env.example` to `.env` only if you want the language-model layer.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GOOGLE_API_KEY` | No | — | Gemini narrative feedback. Absent, the app runs its hybrid fallback. |
| `KEYWORD_WEIGHT` | No | `0.4` | Read into `Config`, currently unused by the scorer |
| `SEMANTIC_WEIGHT` | No | `0.6` | Read into `Config`, currently unused by the scorer |
| `MIN_RELEVANCE_SCORE` | No | `30` | Threshold below which a resume is marked not relevant |
| `MAX_FILE_SIZE_MB` | No | `10` | Upload size limit |
| `ALLOWED_EXTENSIONS` | No | `pdf,docx` | Accepted upload types |
| `LANGCHAIN_API_KEY` | No | — | LangSmith tracing, off by default |

### Running

```bash
.venv/Scripts/python -m streamlit run app/main.py
```

The app serves on `http://localhost:8501`. Load the sample JD from the sidebar,
save it, then upload a resume in step 2.

## Project structure

```
code4ed-resume-analyser/
├── app/main.py           # Streamlit entry point — the three-step flow
├── app/services/         # parsers, keyword_matcher, scorer, vector_store, llm_service
├── app/pages/            # additional Streamlit pages
├── app/models/           # data models for evaluations and job descriptions
├── config/               # application settings
└── test_*.py             # unit tests for matcher, scorer, text processing, vector store
```

## Limitations

- **No semantic retrieval.** The vector store is disabled, so two documents that
  describe the same skill in different words are matched only by fuzzy string
  similarity or by the LLM. This is the most consequential gap between what the
  project set out to do and what it does.
- **No accuracy evaluation.** There is no labelled set of resume/JD pairs with
  ground-truth relevance in this repository, so the rubric weights are reasoned
  defaults rather than tuned ones. No accuracy figure is claimed because none has
  been measured.
- **English only.** Parsing and matching assume English resumes and descriptions.
- **Skill extraction is vocabulary-bound.** A skill the extractor does not know
  cannot be reported as present or missing.
- **Formatting-sensitive parsing.** Multi-column and heavily designed resume
  templates degrade text extraction, which degrades everything downstream.
- **Not bias-audited.** The system has not been tested for differential behaviour
  across candidate demographics, which is a prerequisite for real screening use.
- **Hackathon scope.** Built in the challenge window; there is no authentication,
  no multi-tenancy and no audit trail on decisions.

## Roadmap

- Re-enable semantic matching with an embedding backend that survives the deploy
  target, and actually consume `KEYWORD_WEIGHT` / `SEMANTIC_WEIGHT`
- A labelled evaluation set so the rubric weights can be tuned rather than assumed
- Bias audit across name, gender and institution signals before any real screening use
- Structured section parsing (education, experience, projects) rather than flat text
- Explanation of *why* each requirement matched, surfaced next to the score
- Export a batch ranking as CSV for placement-cell workflows

## Team

Built for the Code4Edtech Challenge by Innomatics Research Labs, as Team F8 —
**V Varadharajan** and **A Sowmiya Priya**. Placed **36th of 1037 teams**.
