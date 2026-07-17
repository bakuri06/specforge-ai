# SpecForge AI

Intelligent Requirements-to-Test State Machine. A local, multi-model agent network
that turns raw requirements (text/PDF/CSV/screenshots) into a polished spec, a QA
test strategy matrix, and export-ready test artifacts (TestRail Markdown, qTest
CSV, Playwright TS stub).

## Architecture

```
[Raw Inputs] --> [qwen2.5vl:7b]     Visual DOM/element extraction (images only)
             --> [deepseek-r1:7b]   Agent 1: BA Requirements Refiner (+ clarification loop)
             --> [deepseek-r1:7b]   Agent 2: QA Test Matrix Builder (+ gap clarifier loop)
             --> [qwen2.5:7b]       Agent 3: Formatter Router (TestRail / qTest / Playwright)
```

These are just the defaults — the Upload step shows a model selector (backed
by `GET /api/models`, which lists whatever's actually pulled in your local
Ollama) so each session can use different models per role without touching
config.

Orchestration is a LangGraph state machine with two human-in-the-loop interrupt
points (ambiguity clarification, test-gap clarification) plus a manual checklist
edit step before formatting. See `backend/app/graph/`.

## Repo layout

```
backend/    FastAPI + LangGraph orchestration, Ollama client, file parsers
frontend/   React (Vite) + Tailwind multi-step wizard UI
```

## Prerequisites

- [Ollama](https://ollama.com) running natively on the host with at least these models pulled:
  ```
  ollama pull qwen2.5vl:7b
  ollama pull deepseek-r1:7b
  ollama pull qwen2.5:7b
  ```
  Pull additional/larger models (e.g. `deepseek-r1:14b`, `qwen2.5-coder:14b`)
  too if you want them selectable — the Upload step's model dropdowns list
  whatever `ollama list` shows, nothing is hardcoded on the frontend.
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

## Troubleshooting

Real issues hit while setting this up, roughly in the order you're likely to hit them:

**`command not found: pip` (macOS)**
You're outside an activated venv, or calling bare `pip`/`python` where only
`pip3`/`python3` exist. See Setup above — activate `.venv` first.

**CORS error / 404 in the browser console when submitting the Upload step**
Usually not actually a CORS problem — the frontend is pointed at the wrong
backend port. It defaults to `http://localhost:8000`; if you started the
backend on a different port (e.g. `--port 8001`), create `frontend/.env` with
`VITE_API_BASE_URL=http://localhost:8001` and **restart** `npm run dev` (Vite
only reads `.env` at startup, a running dev server won't pick up the change).

**`TypeError: unsupported operand type(s) for |: ...` on backend startup**
Python <3.10 doesn't support `X | None` type-hint syntax. The codebase targets
Python 3.9+ and uses `typing.Optional`/`Union` instead — if this reappears,
someone reintroduced `X | Y` syntax somewhere.

**`RuntimeError: Called get_config outside of a runnable context`**
Looks like a LangGraph bug; the actual cause we found was running against
**macOS's system Command Line Tools Python**
(`/Library/Developer/CommandLineTools/usr/bin/python3`) instead of an
isolated project venv — its shared user site-packages accumulate dependencies
from every other project on the machine, so even reinstalling
`requirements.txt` doesn't give a clean dependency set. Check this first,
before anything else, whenever the LangGraph pipeline misbehaves:
```
python3 -c "import sys; print(sys.executable)"
```
This must print a path inside `backend/.venv/...`. If it doesn't, rebuild the
venv with a real Python 3.11 (`brew install python@3.11` if you don't have
one), not whatever `python3` resolves to by default:
```
cd backend
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
To confirm LangGraph's `interrupt()` mechanism itself works in your
environment, independent of this app's code, run
`python3 backend/scripts/repro_interrupt.py` — it should print `SUCCESS:
interrupt()/resume works in this environment.` If it doesn't, the problem is
in the Python/langgraph installation, not in SpecForge AI.

**Uploaded legacy CSV lands in the requirements text instead of legacy test cases**
Fixed as of commit `701343c` — use the dedicated "upload a legacy CSV suite"
file input (or the `legacy_files` API field), not the general attachments
input.

## Current state

This is the Shift-1 environment bootstrap: repo structure, FastAPI skeleton with a
working LangGraph graph (state schema + the three agent nodes + both interrupt
points wired), and a React wizard shell with four steps (Upload, Spec Review /
Clarify, Checklist Editor, Export). Node logic currently calls Ollama for real but
prompts are minimal placeholders — Shift 2 fleshes out the actual BA/QA prompt
engineering and delta-analysis logic per the battle plan.
