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
