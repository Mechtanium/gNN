"""The Delta-PINN app page: settings and the deck on the left, the reservoir in
3-D on the top right, the loss curves below it — the ResInsight layout.

Everything the page shows comes from the messages ``workflow.train`` streams:
the ``grid`` message gives the 3-D view its cells and fixed colour ranges, each
``loss`` message extends the curves, each ``state`` snapshot recolours the cells.
The view is bound to the run once; moving a slice slider or picking a field only
changes what is drawn, never the run.
"""

import perdlit as pl

from workflow import train

FIELDS = ["S_w", "S_o", "S_g", "p_o", "p_w", "p_g", "R_so"]
FIELD_LABELS = {
    "S_w": "Water saturation S_w", "S_o": "Oil saturation S_o", "S_g": "Gas saturation S_g",
    "p_o": "Oil pressure p_o [psia]", "p_w": "Water pressure p_w [psia]",
    "p_g": "Gas pressure p_g [psia]", "R_so": "Dissolved gas R_so [Mscf/stb]",
}
PRESETS = {"Quick (CPU, deck-sized)": "quick", "Gold (GPU, notebook config)": "gold"}


def grid_dims(run):
    """The (ni, nj, nk) of the grid message the run has produced, or None."""
    for item in run.outputs:
        if isinstance(item, (list, tuple)) and item and item[0] == "grid":
            try:
                import json
                return tuple(json.loads(item[-1])["dims"])
            except Exception:  # noqa: BLE001 - a malformed frame must not break the page
                return None
    return None


pl.title("Delta-PINN")
pl.markdown(
    "A physics-informed neural network for **reservoir history matching**: the black-oil "
    "balances of a field are enforced through a spectral (Laplace-eigenfunction) residual "
    "while the simulator's cell states and well observations supervise it. Upload an Eclipse "
    "deck, press **Train**, and watch the losses fall and the cell states form on the grid."
)

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
        retain={"kind_field": 0, "keep": {"state": n_snapshots, "grid": 64, "stage": 8}},
        help="Runs OPM Flow, extracts the case through ResInsight, then trains as a job on this "
             "workstation; a deck must be uploaded first.",
    )

    pl.header("View")
    field = pl.selectbox("Field", FIELDS, index=0)
    t_index = pl.slider("Snapshot (report time index)", 0, max(0, n_snapshots - 1), max(0, n_snapshots - 1))
    z_scale = pl.slider("Z scale (vertical exaggeration)", 1, 20, 5)
    dims = grid_dims(run) or (25, 25, 15)
    with pl.expander("Slice", expanded=True):
        i_lo = pl.slider("i from", 1, dims[0], 1)
        i_hi = pl.slider("i to", 1, dims[0], dims[0])
        j_lo = pl.slider("j from", 1, dims[1], 1)
        j_hi = pl.slider("j to", 1, dims[1], dims[1])
        k_lo = pl.slider("k from", 1, dims[2], 1)
        k_hi = pl.slider("k to", 1, dims[2], dims[2])

# ── Right pane, top: the reservoir ──────────────────────────────────────
pl.subheader(FIELD_LABELS[field])
pl.grid3d(
    run, field=field, t_index=t_index,
    i_range=(min(i_lo, i_hi), max(i_lo, i_hi)),
    j_range=(min(j_lo, j_hi), max(j_lo, j_hi)),
    k_range=(min(k_lo, k_hi), max(k_lo, k_hi)),
    z_scale=z_scale, height=560, key="reservoir",
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
