# Deployment configurations — status: PREPARED, NOT VALIDATED.
#
# - deploy/systemd/laya.service : Linux systemd unit (independent Laya).
# - docker-compose.yml (repo root) : laya service, restart: unless-stopped.
# - Native (tested path): scripts/run_laya_server.py --offline supervised
#   by the platform supervisor; backend connects with LAYA_ALLOW_SPAWN=0.
#
# Validation requires the target runtime:
# - systemd: NOT VALIDATED (Windows host; no systemd here).
# - Docker: NOT VALIDATED (Docker runtime unavailable on this host).
# Do NOT claim supervised deployment until executed on the target.
