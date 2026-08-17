from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response

import subprocess
import json
import time
import re

latency_instance_name = 'latency_api_app'


class Network6GMonitor(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(Network6GMonitor, self).__init__(*args, **kwargs)

        wsgi = kwargs['wsgi']
        wsgi.register(LatencyController,
                      {latency_instance_name: self})

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):

        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()

        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]

        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions)]

        mod = parser.OFPFlowMod(datapath=datapath,
                                priority=0,
                                match=match,
                                instructions=inst)

        datapath.send_msg(mod)


class LatencyController(ControllerBase):

    def __init__(self, req, link, data, **config):
        super(LatencyController, self).__init__(req, link, data, **config)

    @route('latency', '/latency/{src}/{dst}', methods=['GET'])
    def get_latency(self, req, **kwargs):

        src = kwargs['src']
        dst = kwargs['dst']

        ip_map = {
            "URLLC": "20.0.0.1",
            "eMBB": "20.0.0.2",
            "mMTC": "20.0.0.3",
            "MNR_SVR" : "18.0.0.1"
        }

        latency = None
        loss = None
        error = None

        dst_ip = ip_map.get(dst)
        if dst_ip is None:
            error = "unknown dst '%s' (known: %s)" % (
                dst, ", ".join(sorted(ip_map)))

        pid = None
        if error is None:
            # Anchor the pattern. Unanchored, 'mininet:G6_D1' also matches
            # 'mininet:G6_D10' and pgrep returns several PIDs; the old code
            # silently took the first, which could be the wrong host.
            pid_cmd = "pgrep -f 'mininet:%s$'" % src
            try:
                pid_out = subprocess.check_output(pid_cmd, shell=True)
            except subprocess.CalledProcessError:
                pid_out = ""          # pgrep exits 1 when nothing matches
            except OSError as exc:
                pid_out = ""
                error = "could not run pgrep: %s" % exc

            if error is None:
                pids = [p for p in pid_out.strip().split('\n') if p]
                if not pids:
                    error = ("no Mininet host process matches "
                             "'mininet:%s$' -- is the fabric up?" % src)
                elif len(pids) > 1:
                    error = "'mininet:%s$' matched %d PIDs: %s" % (
                        src, len(pids), ",".join(pids))
                else:
                    pid = pids[0]

        if error is None:
            ping_cmd = "mnexec -a %s ping -c 4 %s" % (pid, dst_ip)
            run_error = None
            try:
                out = subprocess.check_output(
                    ping_cmd, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as exc:
                # ping exits 1 on total loss, but its stdout still carries the
                # summary -- that output is a measurement, not a failure.
                out = exc.output or ""
                run_error = "ping exited %d" % exc.returncode
            except OSError as exc:
                out = ""
                run_error = "could not run mnexec: %s" % exc

            rtt = re.search(
                r'rtt min/avg/max/mdev = [\d\.]+/([\d\.]+)/', out)

            if rtt:
                latency = rtt.group(1)

            loss_match = re.search(
                r'(\d+)% packet loss', out)

            if loss_match:
                loss = loss_match.group(1)

            if loss is None:
                # No summary line at all: the ping never measured anything, so
                # this is a genuine failure and must not look like a null
                # reading. Carry the output so the cause is visible to callers.
                tail = out.strip()[-300:] or "(no output)"
                error = "%s; output: %s" % (
                    run_error or "ping produced no summary line", tail)

        epoch_time = time.time()

        readable_time = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(epoch_time)
        )

        ms = int((epoch_time % 1) * 1000)

        timestamp = "%s.%03d" % (readable_time, ms)

        # `error` is null on a good read. It exists so that a null latency_ms
        # caused by a broken probe is distinguishable from one caused by a
        # host that simply did not answer (loss=100, error=null).
        body = json.dumps({
            "src": src,
            "dst": dst,
            "latency_ms": latency,
            "packet_loss_percent": loss,
            "error": error,
            "timestamp": timestamp,
            "epoch": epoch_time
        })

        return Response(content_type='application/json', body=body)
