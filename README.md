# SpecForge AI

Intelligent Requirements-to-Test State Machine. A local, multi-model agent network
that turns raw requirements (text/PDF/CSV/screenshots) into a polished spec, a QA
test strategy matrix, and export-ready test artifacts (TestRail Markdown, qTest
CSV, Playwright TS stub).

## Architecture

```
[Raw Inputs] --> [qwen2.5vl:7b]  Visual DOM/element extraction (images only)
             --> [deepseek-r1:14b]  Agent 1: BA Requirements Refiner (+ clarification loop)
             --> [deepseek-r1:14b]  Agent 2: QA Test Matrix Builder (+ gap clarifier loop)
             --> [qwen2.5-coder:14b] Agent 3: Formatter Router (TestRail / qTest / Playwright)
```

Orchestration is a LangGraph state machine with two human-in-the-loop interrupt
points (ambiguity clarification, test-gap clarification) plus a manual checklist
edit step before formatting. See `backend/app/graph/`.

## Repo layout

```
backend/    FastAPI + LangGraph orchestration, Ollama client, file parsers
frontend/   React (Vite) + Tailwind multi-step wizard UI
```

## Prerequisites

- [Ollama](https://ollama.com) running natively on the host with the three models pulled:
  ```
  ollama pull qwen2.5vl:7b
  ollama pull deepseek-r1:14b
  ollama pull qwen2.5-coder:14b
  ```
- Python 3.9+ and Node 20+.

## Setup

**Backend**
```
cd backend
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env        # adjust OLLAMA_BASE_URL if needed
uvicorn app.main:app --reload --port 8000
```
If `python3`/`pip` aren't found at all, Python itself isn't installed —
macOS: `brew install python@3.11`. Once the venv is activated (prompt shows
`(.venv)`), `pip` refers to the venv's own pip; there's no need for a
system-wide `pip`/`pip3`.

**Frontend**
```
cd frontend
npm install
npm run dev
```

The frontend expects the backend at `http://localhost:8000` (see `frontend/.env`).

## Current state

This is the Shift-1 environment bootstrap: repo structure, FastAPI skeleton with a
working LangGraph graph (state schema + the three agent nodes + both interrupt
points wired), and a React wizard shell with four steps (Upload, Spec Review /
Clarify, Checklist Editor, Export). Node logic currently calls Ollama for real but
prompts are minimal placeholders — Shift 2 fleshes out the actual BA/QA prompt
engineering and delta-analysis logic per the battle plan.
