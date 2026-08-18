#!/usr/bin/env python
"""Bring up the 6G fabric headless and hold it up until killed.

Python 2.7 -- runs inside the Mininet/Ryu container, not in the host venv.

`mn --custom ... --topo sixg` needs a terminal: its CLI reads stdin, and on EOF
it returns and tears the whole network down. That makes it unusable from a
detached `docker exec`, which is the only way to rebuild the fabric without a
human typing into Terminal 1. This builds the same SixGTopo through Mininet's
API, starts it, and then blocks instead of opening a CLI, so the namespaces
persist for as long as the process lives.

Equivalent to:
    mn --custom /root/topo_6g.py --topo sixg \
       --controller=remote,ip=127.0.0.1,port=6633 \
       --switch=ovs,protocols=OpenFlow13

Run (inside the container, AFTER `mn -c`, BEFORE starting ryu-manager):
    docker exec -d <container> sh -c \
        'python /root/run_topo.py >> /tmp/mn.log 2>&1'

Stop it with SIGTERM (`pkill -f run_topo.py`); net.stop() runs on the way out so
the veths and namespaces are cleaned up rather than left dangling for `mn -c`.
"""

import os
import signal
import subprocess
import sys
import time
from functools import partial

from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.log import setLogLevel, info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topo_6g import SixGTopo

_running = [True]

# Upper bound on net.stop(). Generous for a healthy teardown (a few seconds),
# short enough that a wedged one cannot outlive a rebuild.
STOP_TIMEOUT_S = 60


def _sysctl_root(setting):
    """Apply a sysctl in the container's root namespace (where switch veths live)."""
    subprocess.call(["sysctl", "-qw", setting])


def _stop(signum, _frame):
    info("*** caught signal %d, stopping network\n" % signum)
    _running[0] = False


class _StopTimeout(Exception):
    pass


def _stop_net(net, timeout=STOP_TIMEOUT_S):
    """net.stop(), but bounded so a wedged teardown cannot hang the process.

    If the fabric was destroyed underneath us -- someone ran `mn -c`, or started
    a second Mininet -- net.stop() blocks indefinitely trying to tear down veths
    and namespaces that no longer exist, and the process then ignores SIGTERM and
    needs a SIGKILL. Observed on 2026-08-18: this held a dead network for 20+
    minutes alongside a live one, which is dangerous because a late-completing
    net.stop() can delete the *replacement* fabric's bridges.

    SIGALRM interrupts the blocking call. Teardown may be left half-finished, so
    say so plainly: `mn -c` is the cleanup.
    """
    def _on_alarm(_signum, _frame):
        raise _StopTimeout()

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout)
    try:
        net.stop()
        info("*** network stopped\n")
    except _StopTimeout:
        info("*** net.stop() still running after %ds -- abandoning teardown.\n"
             "*** Leftover veths/namespaces are likely; run `mn -c`.\n" % timeout)
    except Exception as exc:  # teardown must never block process exit
        info("*** net.stop() failed: %s\n*** Run `mn -c`.\n" % exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main():
    setLogLevel("info")

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    net = Mininet(
        topo=SixGTopo(),
        # protocols= must be forced here: OVS defaults to OpenFlow10, and every
        # flow this project installs is OpenFlow13.
        switch=partial(OVSSwitch, protocols="OpenFlow13"),
        controller=partial(RemoteController, ip="127.0.0.1", port=6633),
        # Both left at `mn`'s own defaults so this is a drop-in replacement:
        # `mn` sets MACs only under --mac, and waiting for connections is
        # pointless here because ryu-manager is started *after* the fabric.
        autoSetMacs=False,
        waitConnected=False,
    )

    # Kill IPv6 on the switch veths (root namespace) BEFORE the fabric starts
    # forwarding. Every veth otherwise emits router solicitations to ff02::2,
    # which are not IPv4 and so fall through qos_rest_router's
    # `table=1 priority=0 actions=NORMAL` rule and get L2-flooded. The fabric
    # has a physical loop (core-sp1-lf1-sp2-core), there is no spanning tree,
    # and flooding never decrements the IPv6 hop limit -- so the solicitations
    # circulate and replicate until both paths sit at link capacity (measured:
    # 17,835 pkt/s, 10.0 Mbps of pure junk on core-eth2). Once seeded the storm
    # is self-sustaining, and disabling IPv6 afterwards does NOT clear it, so
    # this has to happen while ryu-manager is still down and the switches are
    # dropping everything in fail_mode=secure.
    _sysctl_root("net.ipv6.conf.all.disable_ipv6=1")
    _sysctl_root("net.ipv6.conf.default.disable_ipv6=1")

    net.start()

    for host in net.hosts:
        # Fresh network namespaces do not inherit the root namespace's sysctls.
        host.cmd("sysctl -qw net.ipv6.conf.all.disable_ipv6=1")
        # Hosts come up with only their on-link /24 route, so every
        # cross-subnet ping fails with "Network is unreachable" and /latency
        # reports null until a default gateway exists. Each subnet's router
        # address is x.x.x.254 (prod.json startup_setup.routing: RAN owns
        # 10.0.0.254 and 17.0.0.254, lf1 owns 20.0.0.254 and 18.0.0.254).
        gateway = host.IP().rsplit(".", 1)[0] + ".254"
        host.cmd("ip route replace default via %s" % gateway)
        info("***   %s default via %s\n" % (host.name, gateway))

    info("*** switches: %s\n" % " ".join(s.name for s in net.switches))
    info("*** hosts: %s\n" % " ".join(h.name for h in net.hosts))
    for host in net.hosts:
        info("***   %s pid=%d %s\n" % (host.name, host.pid, host.IP()))
    info("*** fabric up, holding (SIGTERM to stop)\n")
    sys.stdout.flush()

    try:
        while _running[0]:
            time.sleep(1)
    finally:
        _stop_net(net)

    return 0


if __name__ == "__main__":
    sys.exit(main())
