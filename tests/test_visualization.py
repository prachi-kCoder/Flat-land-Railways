"""Small tests for the dependency-free HTML replay output."""
from visualization.replay import _render_html


def test_replay_html_embeds_data_and_controls():
    payload = {
        "frames": [{"step": 0, "trains": [], "delay": 0}],
        "geometry": {"height": 1, "width": 1, "edges": [], "cells": []},
        "summary": {
            "policy": "mappo", "seed": 42, "trains": 1, "steps": 0,
            "arrived": 0, "final_delay": 0, "deadlocked": 0,
            "disruption_plan": [],
        },
        "initial_trains": [],
        "events": [],
        "decisions": [],
        "delay_curve": [],
        "reference_curve": None,
        "recovery_steps": None,
    }
    html = _render_html(payload)
    assert "const DATA =" in html
    assert "Railway network" in html
    assert 'id="timeline"' in html
    assert "Policy decisions" in html
    assert "Live simulation status" in html
    assert "Final episode results" in html
    assert "Hollow marker = waiting at start" in html
    assert 'id="decisionSummary"' in html
    assert 'id="onTrack"' in html


def test_replay_html_escapes_script_closing_sequences():
    payload = {
        "message": "</script><script>alert(1)</script>",
    }
    html = _render_html(payload)
    assert "</script><script>alert(1)</script>" not in html
    assert "<\\/script>" in html


def test_compare_html_has_shared_timeline_and_results_table():
    from visualization.replay import _render_compare_html

    run = {
        "key": "fcfs", "name": "FCFS", "description": "Rule", "colour": "#338ced",
        "frames": [{"step": 0, "trains": [], "delay": 0, "arrived": 0, "on_track": 0, "malfunctioning": 0}],
        "events": [], "decisions": [], "delay_curve": [],
        "summary": {"policy": "fcfs", "seed": 1, "trains": 1, "steps": 0, "arrived": 0,
                    "final_delay": 0, "deadlocked": 0, "disruption_plan": []},
        "extra_delay": None, "recovery_steps": None,
    }
    payload = {"trains": 1, "seed": 1, "disruption_plan": [],
               "geometry": {"height": 1, "width": 1, "edges": [], "cells": []},
               "initial_trains": [], "runs": [run, dict(run, key="priority", name="Priority")]}
    html = _render_compare_html(payload)
    assert "Approach Comparison" in html
    assert html.count('id="timeline"') == 1
    assert 'id="delayChart"' in html
    assert 'id="results"' in html
    assert "Extra delay from breakdowns" in html


def test_compare_runs_every_approach_on_the_same_scenario(tmp_path):
    import json
    import sys
    from unittest import mock

    from visualization import replay

    out = tmp_path / "compare.html"
    argv = ["visualize.py", "--compare", "fcfs", "priority", "--trains", "4", "--seed", "3",
            "--disruptions", "1", "--duration", "5", "--output", str(out), "--no-open"]
    with mock.patch.object(sys, "argv", argv):
        replay.main()
    html = out.read_text(encoding="utf-8")
    data = json.loads(html.split("const DATA = ", 1)[1].split(";\nconst TRAIN_COLOURS", 1)[0])
    assert [r["name"] for r in data["runs"]] == ["FCFS", "Priority"]
    plans = [r["summary"]["disruption_plan"] for r in data["runs"]]
    assert plans[0] == plans[1] == data["disruption_plan"]
    assert all(r["summary"]["trains"] == 4 for r in data["runs"])
