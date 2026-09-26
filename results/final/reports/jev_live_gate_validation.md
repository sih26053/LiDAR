# Jev Live Gate Validation (2026-09-25, post-fix update)

Pre-fix: BLOCKED ("Jev credentials not configured") -> key pasted to
config/local_secrets.py (git-ignored) -> READY since run-20260925T101257.

## Current gate state — READY (live, just re-verified)
check_jev.py: available=true, key_source=config.local_secrets,
endpoint=https://openrouter.ai/api/alpha/decisions (source=default),
model=typesafe/jev-1.13. Probe uses the production enriched state.
Transport (mock-verified): {model, state, questions} POST; bounded
Choice; no chat-completions; no jev-latest; Bearer header; no key in
logs/records (leak tests in suite, 226 passed).

## Live-call ledger post-key (all measured, all archived)
- run-20260925T101257: 10/10 OK (pre-fix state/criteria; all STOP)
- Phase D probes: 6/6 OK (post-fix; 5 directional + 1 stop)
- Phase E loop: 6/6 OK (post-fix; all forward)
- Phase F scenarios: 24/24 OK (post-fix; directional + stops)
- check_jev probes: READY each run
Total: 46+ successful live Decisions-API calls, 0 FAILED, 0 INVALID.

## Key hygiene
Key exists ONLY in config/local_secrets.py (git check-ignore
verified). Never in frontend, logs, APIs, CSVs, reports, or git.
