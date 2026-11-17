"""Run the gNN app on this machine: a worker, an orchestrator and the page, in one process.

    .venv/bin/python tools/local_app.py [--port 8501]

then open http://127.0.0.1:8501/. It is the shape a PERD workstation has — the page's buttons
start jobs on the orchestrator, which runs them on the worker — wired on loopback with no
control plane and no tokens (the orchestrator's development mode). Jobs run in this process's
worker threads; ``GNN_WORK`` sets where the models' work directories go (``/tmp/gnn``), and
ResInsight is found through ``RESINSIGHT_EXECUTABLE`` as the workflow always finds it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _serve(app, port: int, stop: asyncio.Event):
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.bind = [f"127.0.0.1:{port}"]
    config.errorlog = None
    config.accesslog = None
    config.graceful_timeout = 2.0
    return asyncio.create_task(serve(app, config, shutdown_trigger=stop.wait))


async def main(port: int) -> None:
    import grpc

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    import workflow
    from perd import Workstation
    from perd.v1 import job_connect, worker_pb2_grpc
    from perd_orchestrator import InMemoryJobStore, Orchestrator, WorkerPool
    from perd_worker.compiler import compile_contract
    from perd_worker.runtime import WorkerServicer
    from perdlit._runs import Dispatcher
    from perdlit._script import ScriptRunner
    from perdlit._server import PageApp

    registry = workflow.workflow
    compiled = compile_contract("gnn", registry)
    servicer = WorkerServicer(compiled, registry)
    contract = servicer._describe.contract

    server = grpc.aio.server(options=[("grpc.max_receive_message_length", 64 << 20),
                                      ("grpc.max_send_message_length", 64 << 20)])
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    worker_port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    pool = WorkerPool()
    await pool.register(f"127.0.0.1:{worker_port}", expected_hash=compiled.contract_hash)
    orchestrator = Orchestrator(InMemoryJobStore(), pool, heartbeat_interval_s=2.0)
    orch_port = _free_port()
    orch_url = f"http://127.0.0.1:{orch_port}"
    stop = asyncio.Event()
    tasks = [await _serve(job_connect.JobServiceASGIApplication(orchestrator), orch_port, stop)]

    dispatcher = Dispatcher(registry, workstation_factory=lambda bearer: Workstation(orch_url, [contract], auth=bearer),
                            runs=None)
    page = PageApp(ScriptRunner("page"), dispatcher, title="gNN", icon_path=str(ROOT / "favicon.ico"))
    tasks.append(await _serve(page, port, stop))
    print(f"gNN is at http://127.0.0.1:{port}/  (worker :{worker_port}, orchestrator :{orch_port}; Ctrl-C stops)",
          flush=True)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    page.close()
    await asyncio.wait(tasks, timeout=5.0)
    await pool.close()
    await server.stop(grace=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8501)
    asyncio.run(main(parser.parse_args().port))
