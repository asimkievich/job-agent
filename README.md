# job-agent

An experimental automation project using Playwright and Python to explore
agentic workflows for job discovery, scraping, and structured extraction.

## Current status

- Playwright smoke test working
- Project scaffolding in place
- Git + GitHub integration complete

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
playwright install
python smoke_test.py