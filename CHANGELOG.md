# Changelog

## The application release

gNN becomes an application: build a reservoir model, hand it to OPM Flow, and run one of three work paths on it.

- **A model builder.** Design a model in the app — grid, structural top, faults drawn on a map, inactive cells, property recipes, regions, fluids from correlations, rock–fluid functions, initial equilibrium, wells along their paths, a schedule of controls — or import any deck tree (Norne included) and edit it keyword by keyword; untouched files are written back byte for byte. Five templates start a project: three designs and the SPE1 and SPE-2 decks.
- **Check and Simulate.** OPM Flow reads the deck and reports its warnings and errors with file and line, and shows the model it built in 3-D, in seconds; a simulation streams its report steps and shows the cell states and every well's record.
- **Three work paths, no settings.** The network is the research campaign's `AUTOPILOT` composition. **History match** learns the simulator's whole history; **Match + forecast** learns it up to the model's history | prediction boundary and carries the field past it with the balances, the controls and the phase closures; **Invert** teaches a family of simulated reservoirs and inverts the field's well record for the vertical permeability and the fluid contacts, with the family's own error folded into the posterior and the result verified by one more simulator run.
- **A desktop layout.** A toolbar of project commands and jobs, an explorer of the model, tabs for each stage of the work, a status bar. Needs perd-worker 0.9.
- **Removed:** the `train` operation and its presets and network-size parameters.
