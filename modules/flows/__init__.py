r"""
The work paths: what the workflow does with a model, each one a fixed sequence of stages.

Every path is a plain synchronous generator of messages (:mod:`modules.frames`) that
takes a bundle (:mod:`modules.flows.common`) and nothing else a user could set:

- :func:`modules.flows.check.run` — build the deck tree and report what is wrong with it;
- :func:`modules.flows.simulate.run` — hand the deck to the simulator and show its result;
- :func:`modules.flows.history.run` — match the whole simulated history;
- :func:`modules.flows.forecast.run` — match the history up to a boundary and forecast past it;
- :func:`modules.flows.inversion.run` — teach a family of reservoirs and invert a field record through it.

The composition they train is :mod:`modules.autopilot`'s; a ``budget`` argument exists for
the test suite alone. The operations of ``workflow.py`` pull each step of a generator on a
worker thread, so the gRPC server keeps answering while a step computes.
"""
