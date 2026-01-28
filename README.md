# Job Agent

This project is an experimental AI-powered job search assistant.

## Current status
- Playwright smoke test working
- Project scaffolding complete

## Next steps
- Add configuration file
- Add first real scraping script

## Setup#
```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
playwright install
python smoke_test.py