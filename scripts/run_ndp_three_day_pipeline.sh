#!/usr/bin/env bash
set -euo pipefail

RUN_TIME=""
OUT_ROOT=""
MAIN_LOG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-time) RUN_TIME="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --main-log) MAIN_LOG="$2"; shift 2 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

if [[ ! "$RUN_TIME" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "RUN_TIME 格式错误：$RUN_TIME" >&2
  exit 2
fi
if [[ -z "$OUT_ROOT" || -z "$MAIN_LOG" ]]; then
  echo "缺少 --out-root 或 --main-log" >&2
  exit 2
fi

mkdir -p /home/DIFFER/logs
mkdir -p "$OUT_ROOT"
printf '%s\n' "$$" > "$OUT_ROOT/后台进程_${RUN_TIME}.pid"
exec >> "$MAIN_LOG" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] 【后台启动】RUN_TIME=$RUN_TIME；运行方式=nohup；日志只写 /home/DIFFER/logs。"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] 【PPU环境】source /usr/local/PPU_SDK/envsetup.sh ppu；禁止 CPU model fallback。"
: "${LD_LIBRARY_PATH:=}"
: "${LIBRARY_PATH:=}"
source /usr/local/PPU_SDK/envsetup.sh ppu >/dev/null 2>&1
export PYTHONUNBUFFERED=1
export NDP_NO_CPU_FALLBACK=1
exec /usr/local/bin/python3 /home/DIFFER/scripts/ndp_three_day_orchestrator.py --run-time "$RUN_TIME" --out-root "$OUT_ROOT" --main-log "$MAIN_LOG"
