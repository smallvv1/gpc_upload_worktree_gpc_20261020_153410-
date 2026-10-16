#!/usr/bin/env bash
set -euo pipefail

RUN_TIME="${1:?RUN_TIME required}"
ORCH_PID="${2:?current orchestrator PID required}"
OUT_ROOT="${3:?out root required}"
MAIN_LOG="${4:?main log required}"
WRAPPER="/home/DIFFER/scripts/run_ndp_three_day_pipeline.sh"
STATE="${OUT_ROOT}/运行状态_${RUN_TIME}.json"
PID_FILE="${OUT_ROOT}/后台进程_${RUN_TIME}.pid"
SUPERVISOR_PID_FILE="${OUT_ROOT}/后台监督进程_${RUN_TIME}.pid"
SUPERVISOR_SCRIPT="/home/DIFFER/scripts/ndp_resume_supervisor.sh"

printf '%s\n' "$$" > "${SUPERVISOR_PID_FILE}"
log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> "${MAIN_LOG}"
}

log "【有限恢复监督器】已启动；只等待当前 orchestrator PID=${ORCH_PID}，不重启正在运行的训练 arm。"
while kill -0 "${ORCH_PID}" 2>/dev/null; do
  sleep 20
done
sleep 3

DECISION=$(/usr/local/bin/python3 - "${STATE}" "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
out_root = Path(sys.argv[2])
try:
    state = json.loads(state_path.read_text(encoding="utf-8"))
except Exception:
    print("NO_RESTART")
    raise SystemExit

status = state.get("status")
stage = str(state.get("current_stage", ""))
restart_count = int(state.get("supervisor_restart_count", 0))
seed_manifests = [
    out_root / "trajectory_runs" / f"seed{seed}" / "trajectory" / f"seed{seed}" / "trajectory_manifest.json"
    for seed in (1234, 4321)
]
terminal = {"COMPLETED", "STOPPED_BY_SCIENTIFIC_GATE"}
if status == "BLOCKED" and restart_count < 2 and status not in terminal:
    state["supervisor_restart_count"] = restart_count + 1
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("RESTART_ORCHESTRATOR")
else:
    print("NO_RESTART")
PY
)

if [[ "${DECISION}" == "RESTART_ORCHESTRATOR" ]]; then
  log "【有限恢复监督器】检测到 orchestrator 运行阻塞；在同一 RUN_TIME 下启动新版 orchestrator，保留所有失败证据和已有合法产物。"
  nohup setsid bash "${WRAPPER}" \
    --run-time "${RUN_TIME}" \
    --out-root "${OUT_ROOT}" \
    --main-log "${MAIN_LOG}" \
    </dev/null >/dev/null 2>&1 &
  NEW_PID=$!
  printf '%s\n' "${NEW_PID}" > "${PID_FILE}"
  log "【有限恢复监督器】新版 orchestrator 已脱离会话启动；PID=${NEW_PID}；继续交接有限监督。"
  nohup setsid bash "${SUPERVISOR_SCRIPT}" \
    "${RUN_TIME}" "${NEW_PID}" "${OUT_ROOT}" "${MAIN_LOG}" \
    </dev/null >/dev/null 2>&1 &
  CHILD_SUPERVISOR_PID=$!
  log "【有限恢复监督器】子监督器已启动；PID=${CHILD_SUPERVISOR_PID}；最多再允许一次 orchestrator 技术恢复。"
else
  log "【有限恢复监督器】当前退出状态不满足安全恢复条件（${DECISION}）；不擅自重启，保留状态和失败证据。"
fi
rm -f "${SUPERVISOR_PID_FILE}"
log "【有限恢复监督器】退出；不存在永久监控循环。"
