"""Interactive Streamlit + Plotly dashboard that replays JSONL run logs.

Modules:
    replay: Pure functions that load run logs and derive what the views show
        at a simulation minute ``t`` (snapshots, agent status, negotiation
        threads, KPIs).
    figures: Pure Plotly figure builders (map, resources, timelines, KPIs).
    app: The thin Streamlit app (``streamlit run dashboard/app.py``).

The dashboard depends only on the log contract (``emergency_sim.domain``),
``emergency_sim.eventlog``, ``emergency_sim.geo`` and, optionally,
``emergency_sim.evaluation``. It never runs the engine or calls a model.
"""
