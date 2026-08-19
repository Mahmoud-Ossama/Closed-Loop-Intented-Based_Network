#!/usr/bin/env bash
#
# Start ryu-manager in the Mininet/Ryu container, writing its log to a path on
# the HOST rather than to the container's /tmp.
#
# Why: /tmp/ryu.log lives in the container's writable layer. It survives a
# container restart but nothing tells you the controller behind it has died --
# the file just stops growing, and a `tail` of it still prints plausible-looking
# recent activity. On 2026-08-18 that cost an investigation: a tail showing one
# switch mid-ARP-cycle was read as live behaviour when the log had been frozen
# for 1h44m and the whole fabric was gone. With the log on the host, `ls -l`
# answers "is this current?" without entering the container.
#
# The container cannot simply be given a bind mount: adding one requires
# recreating it, and its writable layer holds ryu/app/qos_rest_router.py, which
# is NOT in the osrg/ryu-book image. Recreating would destroy the app the whole
# project depends on. So the log is redirected host-side instead: ryu-manager
# runs via `docker exec` (not `-d`), and its stdout/stderr land directly in a
# host file.
#
# PYTHONUNBUFFERED=1 matters. Without a TTY, python2 block-buffers stdout, so
# the log would lag minutes behind reality and its mtime would lie -- defeating
# the entire point.
#
# Usage:
#   ./scripts/start_ryu.sh          # start (refuses if already listening)
#   ./scripts/start_ryu.sh stop
#   ./scripts/start_ryu.sh status
#
# Env:
#   CONTAINER   container running Mininet+Ryu (default: debc86f1904f)
#   RYU_LOG     host path for the log (default: <repo>/logs/ryu.log)

set -euo pipefail

CONTAINER="${CONTAINER:-debc86f1904f}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RYU_LOG="${RYU_LOG:-$REPO/logs/ryu.log}"

RYU_APPS="ryu.app.rest_qos ryu.app.rest_conf_switch ryu.app.qos_rest_router \
ryu.app.rest_topology ryu.app.ofctl_rest network6G_monitor.py ovs_utilization.py"

controller_url() {
    ip="$(docker inspect -f '{{.NetworkSettings.IPAddress}}' "$CONTAINER")"
    echo "http://${ip}:8080"
}

case "${1:-start}" in
stop)
    docker exec "$CONTAINER" sh -c 'fuser -k 8080/tcp 2>/dev/null; pkill -f ryu-manager' \
        >/dev/null 2>&1 || true
    sleep 1
    echo "Ryu stopped"
    exit 0
    ;;
status)
    echo "=== container process ==="
    docker exec "$CONTAINER" pgrep -af ryu-manager || echo "  not running"
    echo "=== listener ==="
    docker exec "$CONTAINER" sh -c 'ss -lntp 2>/dev/null | grep 8080' || echo "  nothing on 8080"
    echo "=== host log ==="
    if [ -f "$RYU_LOG" ]; then
        ls -la --time-style=full-iso "$RYU_LOG"
        echo "  age: $(( $(date +%s) - $(stat -c %Y "$RYU_LOG") ))s since last write"
    else
        echo "  $RYU_LOG does not exist"
    fi
    exit 0
    ;;
esac

# A stale listener silently makes the new controller useless: ryu-manager exits
# with "Address already in use" but the old, misconfigured process keeps
# answering, so every check below passes against the wrong controller.
if docker exec "$CONTAINER" sh -c 'ss -lntp 2>/dev/null | grep -q 8080'; then
    echo "Error: something already listens on 8080 in $CONTAINER." >&2
    echo "Run '$0 stop' first (see RUNNING_GUIDE §3, 'Restarting Ryu')." >&2
    exit 1
fi

mkdir -p "$(dirname "$RYU_LOG")"

{
    echo
    echo "===== ryu-manager start $(date -Is) (host-side log) ====="
} >> "$RYU_LOG"

# Not `docker exec -d`: detaching would send stdout back to the container's
# void and there would be nothing to redirect. nohup keeps it alive when this
# shell exits. `cd /root` because ryu-manager loads the two custom apps by bare
# relative path.
nohup docker exec -e PYTHONUNBUFFERED=1 "$CONTAINER" \
    sh -c "cd /root && exec ryu-manager --observe-links $RYU_APPS" \
    >> "$RYU_LOG" 2>&1 &

echo "ryu-manager starting (host log: $RYU_LOG)"

url="$(controller_url)"
for i in $(seq 1 30); do
    if curl -s -m 2 "${url}/stats/switches" >/dev/null 2>&1; then
        echo "controller up after ${i}s at ${url}"
        exit 0
    fi
    sleep 1
done

echo "Error: controller did not answer at ${url} within 30s; see $RYU_LOG" >&2
exit 1
