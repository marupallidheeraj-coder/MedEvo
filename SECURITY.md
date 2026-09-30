# Security policy

MedFusion-Evo is a research demo. It must not be used with real patient data or for clinical decisions.

## Reporting a vulnerability
Open a private security advisory on the GitHub repository (Security → Report a vulnerability), or email the
maintainers. Please do not open a public issue for security problems. We aim to respond within 7 days.

## What is already in place
| Area | Measure |
|---|---|
| Uploads | Size cap (8 MB), JPG/PNG allow-list checked from the file contents (not the file name), minimum size, decompression-bomb limit (25 MP), full decode inside a guarded block. `backend/security.py`, tested in `tests/test_security.py` |
| Abuse | Per-IP sliding-window rate limit on `/predict-image` (`RATE_LIMIT_PER_MIN`, default 30) |
| CORS | Only the origins in `ALLOWED_ORIGINS` (default: local development). No wildcard by default |
| Model files | Loaded with `torch.load(..., weights_only=True)`, so a tampered `model.pt` cannot run code |
| Errors | Generic messages to the client, details only in server logs. No stack traces are returned |
| Privacy | Images are decoded in memory, never written to disk or logged. Responses send `Cache-Control: no-store` |
| Container | Runs as an unprivileged user with a health check |
| Website | Security headers and a Content-Security-Policy in `vercel.json`. Values shown in the page come from fixed lists or validated numbers |
| Secrets | `.env` and `kaggle.json` are git-ignored. `.env.example` documents the settings. No secrets are stored in the repo |

## Known limits
- The rate limiter is in memory and per process. Behind a proxy all clients may share one IP. Use the platform's
  own rate limiting for real deployments.
- The Content-Security-Policy allows inline scripts because the site is a single HTML file.
- There is no authentication. Do not expose the API to the internet with real patient data.
