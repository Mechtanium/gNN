"""gNN as a PERD workflow: a reservoir model in, and what its work paths make of it out.

Every operation takes one input — the project **bundle**, as a stream of byte chunks:
a self-contained Eclipse ``.DATA`` deck, or a zip of a deck tree with the project's own
files beside it — and streams messages while it works. There is nothing else to pass:
the composition the paths train is fixed (``modules/autopilot.py``), and what a path
needs to know about the model (where its history ends, the record to invert) is part of
the bundle, as a development strategy is part of a simulation case.

==================  ==========================================================
``check``           have OPM Flow read the deck and build the model, without
                    simulating; stream its findings and the static model
``simulate``        hand the deck to OPM Flow; stream the grid, the simulated
                    cell states and every well's record
``history_match``   work path 1: match the whole simulated history
``history_forecast``  work path 2: match the history up to the model's
                    history | prediction boundary and forecast past it
``invert``          work path 3: invert the field's well record for the model's
                    uncertain properties through a family of reservoirs
==================  ==========================================================

Every output item is one message — ``(kind, iteration, total, pde, ic, data, well,
payload_json)`` — see ``modules/frames.py`` for the kinds and their encoding.

Only ``perd_worker`` is imported at module level: the build's describe step and the page
sidecar import this file on a CPU with no JAX device.
"""

from __future__ import annotations

import asyncio
import traceback

from perd_worker import WorkflowStreamInput, WorkflowStreamOutput, workflow

BUNDLE_MAX_BYTES = 256 * 1024 * 1024

Frames = WorkflowStreamOutput[str, int, float, float, float, float, float, str]


async def _bundle(chunks) -> bytes:
    """The bundle, reassembled from its chunks (any size each)."""
    parts: list[bytes] = []
    size = 0
    async for chunk in chunks:
        parts.append(chunk)
        size += len(chunk)
        if size > BUNDLE_MAX_BYTES:
            raise ValueError(f"bundle larger than {BUNDLE_MAX_BYTES // 2**20} MiB")
    data = b"".join(parts)
    if not data.strip():
        raise ValueError("an empty bundle was uploaded")
    return data


async def _stream(flow: str, chunks, deck_name: str):
    """Run one flow of ``modules.flows`` on the uploaded bundle and yield its messages.

    The workstation's accelerator is held for the length of the job (two trainings sharing
    one run more than twice as slowly each). Each step of the flow's generator is minutes
    of numerics, so it is pulled on a thread: the worker's event loop keeps answering
    heartbeats and cancellation. A failure is sent as an ``error`` message before it is
    raised, so the page shows the reason instead of a bare job failure.
    """
    data = await _bundle(chunks)

    import importlib

    from modules import frames
    from modules.flows import common

    module = importlib.import_module(f"modules.flows.{flow}")

    def messages():
        if flow == "check":                    # seconds of CPU, no accelerator: never queued
            yield from module.run(data, deck_name)
            return
        try:
            yield from common.exclusive(what=flow)
            yield from module.run(data, deck_name)
        finally:
            common.release()

    gen = messages()
    sentinel = object()
    try:
        while True:
            item = await asyncio.to_thread(next, gen, sentinel)
            if item is sentinel:
                break
            yield item
    except Exception as exc:
        yield frames.error(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}")
        raise
    finally:
        gen.close()


@workflow.bi_di
async def check(chunks: WorkflowStreamInput[bytes], deck_name: str = "") -> Frames:
    """Check the model: OPM Flow reads the deck tree and builds the model, then stops.

    Streams every warning and error Flow reports (with the file and line they concern), the
    grid as the simulator built it (inactive cells and pore-volume cuts applied), the wells'
    perforated cells, and the static properties — porosity, permeabilities, net-to-gross,
    pore volume, depth and the region numbers. Seconds even for a field-scale model.
    """
    async for item in _stream("check", chunks, deck_name):
        yield item


@workflow.bi_di
async def simulate(chunks: WorkflowStreamInput[bytes], deck_name: str = "") -> Frames:
    """Simulate the model with OPM Flow and stream its result.

    ``chunks`` is the bundle in pieces; ``deck_name`` names the deck inside a zip bundle
    (the only top-level ``.DATA`` file when left empty). Streams the simulator's report
    steps as it runs, then the grid, the seven cell fields at report times spread over the
    history, and every well's record. Later jobs on the same model reuse the run.
    """
    async for item in _stream("simulate", chunks, deck_name):
        yield item


@workflow.bi_di
async def history_match(chunks: WorkflowStreamInput[bytes], deck_name: str = "") -> Frames:
    """Match the whole simulated history of the model (work path 1).

    Simulates the model unless a run of it is cached, then trains the gNN on the
    simulator's cell states at every report step and on the wells' records, with the
    black-oil balances enforced. Streams the loss per iteration, the predicted cell fields
    as they form, and at the end every well's simulated record against the trained
    field's, with the pressure and saturation errors.
    """
    async for item in _stream("history", chunks, deck_name):
        yield item


@workflow.bi_di
async def invert(chunks: WorkflowStreamInput[bytes], deck_name: str = "") -> Frames:
    """Invert the model's observed field record for its uncertain properties (work path 3).

    The bundle's ``observed/record.csv`` is the field's well record (bottom-hole pressures
    and rates per well and day) and ``inversion/prior.json`` the prior widths. The model is
    simulated at sampled values of its uncertain properties — the vertical permeability, and
    the fluid contacts the reservoir holds — a family network is taught all the runs, scored
    on runs it did not see, and searched for the properties whose wells behave as the
    field's did. Streams the runs as they finish, the family's loss, the posterior of every
    property, and the field's record against the family's and the simulator's at the
    posterior mean. The runs and the trained family are cached: inverting another record of
    the same model only searches again.
    """
    async for item in _stream("inversion", chunks, deck_name):
        yield item


@workflow.bi_di
async def history_forecast(chunks: WorkflowStreamInput[bytes], deck_name: str = "") -> Frames:
    """Match the history up to the model's boundary and forecast past it (work path 2).

    The bundle's ``project.json`` carries the history | prediction boundary
    (``history_end_days``). Up to it the gNN is trained on the simulator's cell states and
    well records; past it only on the balances, the schedule's controls and the phase
    closures. Streams as ``history_match`` does; each point of a well's record is marked
    ``history`` or ``forecast``, and the forecast is scored against the simulator's own
    run past the boundary, which the training never saw.
    """
    async for item in _stream("forecast", chunks, deck_name):
        yield item
