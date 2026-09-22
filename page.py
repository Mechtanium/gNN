"""The gNN app page: the deck and the run settings on the left, the reservoir in
3-D on the top right, the loss curves below it — the ResInsight layout.

Everything the page shows comes from the messages ``workflow.train`` streams:
the ``grid`` message gives the 3-D view its cells and fixed colour ranges, each
``loss`` message extends the curves, each ``state`` snapshot recolours the cells.
The left pane is the run: what to train and the button that starts it. What is
*displayed* — property, snapshot, slices, colour map, exaggeration, playback —
belongs to the 3-D view's own hover panel, so looking around costs no rerun and
nothing here has to mirror it.
"""

import perdlit as pl

from workflow import train

FIELD_LABELS = {
    "S_w": "Water saturation (S_w)",
    "S_o": "Oil saturation (S_o)",
    "S_g": "Gas saturation (S_g)",
    "p_o": "Oil pressure (p_o) [psia]",
    "p_w": "Water pressure (p_w) [psia]",
    "p_g": "Gas pressure (p_g) [psia]",
    "R_so": "Dissolved gas ratio (R_so) [Mscf/stb]",
}
PRESETS = {"Quick (CPU, deck-sized)": "quick", "Gold (GPU, notebook config)": "gold"}


# ── Left pane: run settings and the deck ────────────────────────────────
with pl.sidebar:
    pl.header("Deck")
    deck = pl.file_uploader("Eclipse deck (.DATA, self-contained)", type=["DATA", "data"])

    pl.header("Run")
    preset_label = pl.radio("Preset", list(PRESETS))
    preset = PRESETS[preset_label]
    gold = preset == "gold"
    n_iter = pl.number_input("Iterations", 1, 200000, 20000 if gold else 200)
    seed = pl.number_input("Seed", 0, 9999, 30)
    snapshot_every = pl.number_input("Cell-state snapshot every N iterations", 1, 5000, 50 if gold else 10)
    n_snapshots = pl.number_input("Report times per snapshot", 1, 32, 8)
    with pl.expander("Network (0 = the preset's choice)"):
        m_width = pl.number_input("DGM width", 0, 512, 0)
        n_blocks = pl.number_input("DGM depth (blocks)", 0, 16, 0)
        n_eig = pl.number_input("Eigenmodes n_eig", 0, 63, 0)

    run = pl.button(
        "Train",
        callable=train if deck is not None else None,
        args=[[deck.read()] if deck is not None else [], deck.name if deck is not None else "deck.DATA", preset, n_iter, seed],
        kwargs={"m_width": m_width, "n_blocks": n_blocks, "n_eig": n_eig,
                "log_every": 1, "snapshot_every": snapshot_every, "n_snapshots": n_snapshots},
        deadline_s=72 * 3600 if gold else 4 * 3600,
        # Keep several sweeps of snapshots, not just the newest: the 3-D view's
        # iteration playback walks the frames it still holds.
        retain={"kind_field": 0, "keep": {"state": n_snapshots * 12, "grid": 64, "stage": 8}},
        help="Runs OPM Flow, extracts the case through ResInsight, then trains as a job on this "
             "workstation; a deck must be uploaded first.",
    )

# ── Right pane, top: the reservoir ──────────────────────────────────────
# Its hover panel owns the view: property, snapshot, slicing, colour map,
# per-axis exaggeration, grid lines, axes and the two playbacks.
pl.grid3d(
    run,
    field="S_w",
    t_index=max(0, n_snapshots - 1),
    field_labels=FIELD_LABELS,
    z_scale=5,
    height=560,
    key="reservoir",
)

# ── Right pane, bottom: the losses ──────────────────────────────────────
left, right = pl.columns([3, 1])
with left:
    pl.subheader("Losses")
    pl.line_chart(
        run, x=1, y=[2, 3, 4, 5, 6], where=(0, "loss"), y_scale="log",
        legend=["total", "pde", "ic", "data", "well"], labels=["iteration", "loss"], height=340,
    )
with right:
    pl.subheader("Run")
    pl.metric("Stage", run, field="7.stage", where=(0, "stage"))
    pl.progress(run, field=2, where=(0, "stage"), text="pipeline")
    pl.metric("Iteration", run, field=1, where=(0, "loss"))
    pl.metric("Total loss", run, field=2, where=(0, "loss"), fmt=".4f")
    pl.metric("Well loss", run, field=6, where=(0, "loss"), fmt=".4f")
    pl.progress(run, total=n_iter, field=1, where=(0, "loss"))
    pl.write(run)
