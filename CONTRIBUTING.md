# Contributing

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
make install          # dev dependencies
make test             # Python + JavaScript tests
make lint             # ruff
make validate         # clinical model / GA validation report -> docs/validation_report.json
```

Before opening a pull request:
1. Add or update tests for the behaviour you change (`tests/`). Bug fixes need a test that fails without the fix.
2. `make test` and `make lint` must pass. CI runs both, plus a Docker build.
3. Changes that affect the model or its numbers must update `docs/MODEL_CARD.md` and `docs/VALIDATION.md`.
4. Keep the website a single file with no build step, and keep the accessibility basics: labels on every input,
   keyboard access, and text that does not rely on colour alone.
