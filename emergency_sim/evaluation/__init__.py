"""Evaluation layer: metrics, the run/grid runner and the statistical report.

Modules:
    metrics: ``compute_metrics`` and ``kpis_at`` (exact formulas, plan section 6).
    runner: ``RunSpec``, ``run_simulation`` and ``run_grid`` (composition root).
    report: summaries, paired comparisons and the written report.

No re-exports here: import the submodules directly, so that importing
``metrics`` (e.g. from the dashboard) never pulls in the engine or the SDK.
"""
