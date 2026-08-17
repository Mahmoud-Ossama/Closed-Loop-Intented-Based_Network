#!/usr/bin/env bash
#
# Disable IPv6 across the Mininet fabric.
#
# Every host and switch veth comes up with IPv6 enabled and emits router
# solicitations to ff02::2. Those are neither IPv4 nor ARP, so they miss every
# specific flow and fall through to the priority-0 `actions=NORMAL` fallback,
# which floods multicast. The fabric's main/backup paths (core-sp1-lf1 and
# core-sp2-lf1) form a physical loop, there is no spanning tree, and L2
# flooding never decrements the IPv6 hop limit -- so the solicitations
# circulate permanently and amplify. Measured at 36,840 pkt/s, which starves
# the controller round-trips ARP depends on and leaves hosts with INCOMPLETE
# neighbour entries.
#
# Run this AFTER `mn` comes up and BEFORE starting ryu-manager.
#
# Usage:
#   ./scripts/disable_ipv6.sh
#   CONTAINER=abc123 ./scripts/disable_ipv6.sh

set -euo pipefail

CONTAINER="${CONTAINER:-debc86f1904f}"

echo "*** Disabling IPv6 in container root namespace (switch veths)"
docker exec "$CONTAINER" sysctl -qw net.ipv6.conf.all.disable_ipv6=1
docker exec "$CONTAINER" sysctl -qw net.ipv6.conf.default.disable_ipv6=1

echo "*** Disabling IPv6 in each Mininet host namespace"
pids="$(docker exec "$CONTAINER" pgrep -f 'mininet:' || true)"
if [ -z "$pids" ]; then
    echo "Error: no Mininet host processes found in ${CONTAINER}" >&2
    exit 1
fi

count=0
for pid in $pids; do
    name="$(docker exec "$CONTAINER" cat "/proc/${pid}/cmdline" 2>/dev/null \
            | tr '\0' ' ' | grep -o 'mininet:[^ ]*' || echo "pid ${pid}")"
    if docker exec "$CONTAINER" mnexec -a "$pid" \
           sysctl -qw net.ipv6.conf.all.disable_ipv6=1 2>/dev/null; then
        count=$((count + 1))
    else
        echo "  WARN: could not disable IPv6 for ${name} (pid ${pid})" >&2
    fi
done

echo "*** Disabled IPv6 in ${count} namespace(s)"
