"""gNN as a PERD workflow: a reservoir deck in, a trained history-matching
PINN out, with the losses and the cell states streaming while it trains.

The single operation, ``train``, takes an Eclipse ``.DATA`` deck as a stream of
byte chunks and a few run parameters, then runs the reference simulation (OPM
Flow), extracts the grid, states and wells through ResInsight, builds the
gNN pipeline and trains it with the notebook's gold-standard
configuration (or the deck-sized QUICK preset). Every output item is one
message: a ``stage`` update, the ``grid`` geometry, a ``loss`` per iteration, a
``state`` snapshot of the seven cell fields at one report time, or the final
``metrics`` — see ``modules/frames.py`` for the encoding.

Only ``perd_worker`` is imported at module level: the build's describe step and
the page sidecar import this file on a CPU with no JAX device.
"""

from __future__ import annotations

import asyncio
import traceback

from perd_worker import WorkflowStreamInput, WorkflowStreamOutput, workflow

DECK_MAX_BYTES = 64 * 1024 * 1024


@workflow.bi_di
async def train(
    chunks: WorkflowStreamInput[bytes],
    deck_name: str = "deck.DATA",
    preset: str = "gold",
    n_iter: int = 20000,
    seed: int = 30,
    m_width: int = 0,
    n_blocks: int = 0,
    n_eig: int = 0,
    log_every: int = 1,
    snapshot_every: int = 50,
    n_snapshots: int = 8,
) -> WorkflowStreamOutput[str, int, float, float, float, float, float, str]:
    """Train the gNN on one Eclipse deck and stream the run.

    ``chunks`` is the deck file in pieces (any size). ``preset`` is ``"gold"`` (the
    notebook's configuration, meant for a GPU) or ``"quick"`` (the same physics
    sized to the deck under an 800-parameter target; minutes on a CPU). A zero
    ``m_width``/``n_blocks``/``n_eig`` keeps the preset's value. A ``loss``
    message is emitted every ``log_every`` iterations and a set of ``n_snapshots``
    ``state`` messages every ``snapshot_every`` iterations and at the end.
    """
    parts: list[bytes] = []
    size = 0
    async for chunk in chunks:
        parts.append(chunk)
        size += len(chunk)
        if size > DECK_MAX_BYTES:
            raise ValueError(f"deck larger than {DECK_MAX_BYTES // 2**20} MiB")
    deck = b"".join(parts)
    if not deck.strip():
        raise ValueError("an empty deck was uploaded")

    from modules import frames, presets, stream

    base = presets.preset(preset)
    gen = stream.run_training(
        deck, deck_name, base, n_iter=int(n_iter), log_every=int(log_every),
        snapshot_every=int(snapshot_every), n_snapshots=int(n_snapshots),
        seed=int(seed), m_width=int(m_width), n_blocks=int(n_blocks), n_eig=int(n_eig),
    )
    sentinel = object()
    try:
        while True:
            # Each generator step is minutes of numerics; pull it on a thread so the
            # worker's event loop keeps answering heartbeats and cancellation.
            item = await asyncio.to_thread(next, gen, sentinel)
            if item is sentinel:
                break
            yield item
    except Exception as exc:  # the page shows the reason instead of a bare job failure
        yield frames.error(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}")
        raise
    finally:
        gen.close()
