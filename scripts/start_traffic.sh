#!/usr/bin/env bash
#
# Start the training traffic flows without Mininet's CLI.
#
# traffic_runner.py's start_traffic() takes Mininet's `net` object and is meant
# to be called from inside `mininet>`. When the fabric is held up headlessly by
# scripts/run_topo.py there is no CLI and no `net`, so this reproduces the same
# flows through mnexec: three UDP iperf servers on the lf1 hosts and three
# clients on the RAN hosts, same host pairs, ports and default bandwidths.
#
#   G6_D1    -> URLLC 20.0.0.1:5001  @ 6M
#   G6_D2    -> eMBB  20.0.0.2:5002  @ 5M
#   G6_IOT_D -> mMTC  20.0.0.3:5003  @ 3M
#
# 14M total, which fits the 20 Mbps main path.
#
# Usage:
#   ./scripts/start_traffic.sh                 # 1800s, default bandwidths
#   ./scripts/start_traffic.sh 600             # 600s
#   ./scripts/start_traffic.sh 600 8M 6M 3M
#   ./scripts/start_traffic.sh stop
#   ./scripts/start_traffic.sh status

set -euo pipefail

CONTAINER="${CONTAINER:-debc86f1904f}"
MN="$(dirname "$0")/mn.sh"

SERVERS="URLLC:5001 eMBB:5002 mMTC:5003"
CLIENTS="G6_D1:20.0.0.1:5001 G6_D2:20.0.0.2:5002 G6_IOT_D:20.0.0.3:5003"
ALL_HOSTS="URLLC eMBB mMTC G6_D1 G6_D2 G6_IOT_D"

kill_iperf() {
    for h in $ALL_HOSTS; do
        # pkill returns 1 when nothing matched, which is not an error here.
        "$MN" host "$h" pkill -f iperf >/dev/null 2>&1 || true
    done
}

case "${1:-start}" in
stop)
    kill_iperf
    echo "Traffic stopped"
    exit 0
    ;;
status)
    # Listed once, container-wide, NOT per host: `mnexec -a` enters only the
    # network namespace, so every host's pgrep sees the same global process
    # list and a per-host loop just prints all six flows six times.
    echo "=== iperf processes (container-wide) ==="
    docker exec "$CONTAINER" pgrep -af 'iperf ' | grep -v 'sh -c' || echo "none running"
    echo
    for h in $ALL_HOSTS; do
        tail="$(docker exec "$CONTAINER" tail -n 2 "/tmp/${h}_iperf.log" 2>/dev/null || true)"
        printf "=== %s ===\n%s\n" "$h" "${tail:-(no log)}"
    done
    exit 0
    ;;
esac

DURATION="${1:-1800}"
URLLC_BW="${2:-6M}"
EMBB_BW="${3:-5M}"
MMTC_BW="${4:-3M}"

kill_iperf

# Warm ARP BEFORE any iperf. qos_rest_router installs routes as packet-in stubs
# and suspends packets at the controller while it floods an ARP request for the
# next hop. Pointing 14 Mbps at unresolved destinations turns every packet into
# a packet-in and wedges the router -- it then stops answering ARP for every
# switch, stopping traffic does not recover it, and only a Ryu restart (plus a
# re-run of setup_network.py) clears it. See RUNNING_GUIDE.md §8.15.
echo "Warming ARP (quiet network) before starting iperf..."
warm_failed=0
for entry in $CLIENTS; do
    host="${entry%%:*}"
    rest="${entry#*:}"
    dst="${rest%%:*}"
    if "$MN" host "$host" ping -c 3 -W 2 "$dst" >/dev/null 2>&1; then
        printf "  %-10s -> %-10s ok\n" "$host" "$dst"
    else
        printf "  %-10s -> %-10s UNRESOLVED\n" "$host" "$dst"
        warm_failed=1
    fi
done

if [ "$warm_failed" -ne 0 ]; then
    echo "Error: ARP did not resolve for every flow; refusing to start traffic." >&2
    echo "Starting iperf now would wedge qos_rest_router (see §8.15). Fix" >&2
    echo "connectivity first — a Ryu restart + setup_network.py is the usual cure." >&2
    exit 1
fi

for entry in $SERVERS; do
    host="${entry%%:*}"
    port="${entry##*:}"
    pid="$(docker exec "$CONTAINER" pgrep -f "mininet:${host}\$")"
    docker exec -d "$CONTAINER" sh -c \
        "mnexec -a ${pid} iperf -s -u -p ${port} -i 1 > /tmp/${host}_iperf.log 2>&1"
done

sleep 1

for entry in $CLIENTS; do
    host="${entry%%:*}"
    rest="${entry#*:}"
    dst="${rest%%:*}"
    port="${rest##*:}"
    case "$host" in
        G6_D1)    bw="$URLLC_BW" ;;
        G6_D2)    bw="$EMBB_BW" ;;
        G6_IOT_D) bw="$MMTC_BW" ;;
    esac
    pid="$(docker exec "$CONTAINER" pgrep -f "mininet:${host}\$")"
    docker exec -d "$CONTAINER" sh -c \
        "mnexec -a ${pid} iperf -c ${dst} -u -p ${port} -b ${bw} -t ${DURATION} \
         > /tmp/${host}_iperf.log 2>&1"
done

echo "Traffic started: duration=${DURATION}s urllc=${URLLC_BW} embb=${EMBB_BW} mmtc=${MMTC_BW}"
