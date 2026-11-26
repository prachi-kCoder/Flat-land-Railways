"""Create a self-contained, replayable HTML visualization of a Flatland episode.

No visualization dependency is required: the renderer is plain HTML Canvas.
All positions, states, disruption events and policy decisions are recorded from
the live Flatland environment; the browser only replays those recorded facts.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np
from flatland.envs.step_utils.states import TrainState

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, make_env
from railsim.metrics import network_delay, recovery_time
from railsim.train_status import find_deadlocked_trains


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
def _state_name(agent):
    state = agent.state
    return getattr(state, "name", str(state).split(".")[-1])


def _position(agent):
    """Return [row, col] for an on-map train, otherwise None."""
    if not agent.state.is_on_map_state():
        return None
    configuration = agent.current_configuration
    if configuration is None:
        return None
    cell = configuration[0]
    if cell is None:
        return None
    return [int(cell[0]), int(cell[1])]


def _track_geometry(env):
    """Extract directed track connectivity from Flatland's transition map."""
    height, width = env.height, env.width
    edges = set()
    cells = set()
    rail = env.rail
    for row in range(height):
        for col in range(width):
            for incoming in range(4):
                transitions = rail.get_transitions(((row, col), incoming))
                for outgoing, enabled in enumerate(transitions):
                    if not enabled:
                        continue
                    dr, dc = DIRECTIONS[outgoing]
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        a, b = (row, col), (nr, nc)
                        edge = tuple(sorted((a, b)))
                        edges.add(edge)
                        cells.add(a)
                        cells.add(b)
    # Keep isolated/non-standard rail cells visible if the transition API
    # returns no outgoing links for them.
    if not cells:
        grid = getattr(rail, "grid", None)
        if grid is not None:
            for row in range(height):
                for col in range(width):
                    if int(grid[row, col]) != 0:
                        cells.add((row, col))
    return {
        "height": int(height),
        "width": int(width),
        "edges": [[list(a), list(b)] for a, b in sorted(edges)],
        "cells": [list(cell) for cell in sorted(cells)],
    }


def _train_snapshot(agent):
    config = agent.initial_configuration
    start = (
        [int(value) for value in config[0]]
        if config and config[0] is not None
        else None
    )
    target = getattr(agent, "target", None)
    if target is None:
        target_configuration = getattr(agent, "target_configuration", None)
        if target_configuration is not None:
            target = target_configuration[0]
        else:
            targets = getattr(agent, "targets", ())
            target_positions = sorted({
                tuple(configuration[0])
                for configuration in targets
                if configuration[0] is not None
            })
            target = target_positions[0] if target_positions else None
    target = [int(value) for value in target] if target is not None else None
    current = agent.current_configuration if _position(agent) is not None else config
    direction = int(current[1]) if current is not None else None
    return {
        "id": int(agent.handle),
        "state": _state_name(agent),
        "position": _position(agent),
        "direction": direction,
        "start": start,
        "target": target,
        "malfunction": int(agent.malfunction_handler.malfunction_down_counter),
        "arrival_time": (int(agent.arrival_time) if agent.arrival_time is not None else None),
    }


def _snapshot(env, step):
    trains = [_train_snapshot(agent) for agent in env.agents]
    return {
        "step": int(step),
        "delay": int(network_delay(env)),
        "trains": trains,
        "arrived": sum(t["state"] == "DONE" for t in trains),
        "malfunctioning": sum(t["malfunction"] > 0 or "MALFUNCTION" in t["state"] for t in trains),
        "on_track": sum(t["position"] is not None for t in trains),
    }


# Names shown to people, and fixed colours per approach (same in every view).
DISPLAY_NAMES = {
    "fcfs": "FCFS", "priority": "Priority", "edf": "EDF", "spf": "SPF",
    "greedy_select": "Greedy rule selection", "release_all": "Release all",
    "hold_if_blocked": "Hold if blocked", "rl_strategy": "PPO", "ppo": "PPO", "mappo": "MAPPO",
}
DESCRIPTIONS = {
    "fcfs": "Rule: trains due to leave earlier book their route first",
    "priority": "Rule: the train with the least spare time books first",
    "edf": "Rule: earliest latest-arrival time books first",
    "spf": "Rule: shortest remaining trip books first",
    "greedy_select": "Tries four ordering rules at every re-plan, keeps the best plan",
    "release_all": "Releases every ready train at once",
    "hold_if_blocked": "Releases a ready train only if its route is free now",
    "rl_strategy": "Learned (PPO): picks the ordering rule at every re-plan",
    "ppo": "Learned (PPO): picks the ordering rule at every re-plan",
    "mappo": "Learned (MAPPO): each ready train decides Dispatch or Wait",
}
COMPARE_COLOURS = {
    "fcfs": "#338ced", "priority": "#e76624", "greedy_select": "#00a888",
    "rl_strategy": "#d28e05", "ppo": "#d28e05", "mappo": "#e24d8b",
}
DEFAULT_PPO_MODEL = "results/models/rl_strategy_seed1.zip"


def _default_mappo_model():
    final = Path("results/models/mappo_seed1.pt")
    return str(final) if final.exists() else "models/mappo_seed1.pt"


def _seed_torch(seed):
    """MAPPO samples its actions. Seeding torch before each run (as the
    experiment scripts do) makes a replay repeat the evaluated run exactly."""
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def _make_policy(name, model):
    from railsim.policies import FCFSDispatcher, PriorityDispatcher
    from railsim.rl_dispatch import HoldIfBlocked, ReleaseAll
    from railsim.strategy import EDFDispatcher, SPFDispatcher, GreedySelect, RLStrategy

    policies = {
        "fcfs": FCFSDispatcher,
        "priority": PriorityDispatcher,
        "edf": EDFDispatcher,
        "spf": SPFDispatcher,
        "greedy_select": GreedySelect,
        "release_all": ReleaseAll,
        "hold_if_blocked": HoldIfBlocked,
    }
    if name == "mappo":
        from railsim.mappo import MAPPODispatcher
        MAPPODispatcher.model_path = model or _default_mappo_model()
        MAPPODispatcher._actor = None
        MAPPODispatcher._loaded_from = None
        return MAPPODispatcher()
    if name in ("rl_strategy", "ppo"):
        model = model or (DEFAULT_PPO_MODEL if name == "ppo" else None)
        if not model:
            raise ValueError("--model is required for --policy rl_strategy")
        RLStrategy.model_path = model
        RLStrategy._model = None
        RLStrategy._loaded_from = None
        return RLStrategy()
    if name not in policies:
        raise ValueError(f"Unsupported policy: {name}")
    return policies[name]()


def _run_episode(config, disruption_spec, policy_name, model, record=False):
    env, _, _ = make_env(config, disruption_spec)
    policy = _make_policy(policy_name, model)
    policy.reset(env)

    decisions = []
    events = []
    if record and hasattr(policy, "decide"):
        original_decide = policy.decide

        def recorded_decide(current_env, now, handle):
            action = original_decide(current_env, now, handle)
            decisions.append({
                "step": int(now), "train": int(handle),
                "action": "DISPATCH" if int(action) == 1 else "WAIT",
            })
            return action

        policy.decide = recorded_decide

    if record and hasattr(policy, "choose"):
        original_choose = policy.choose

        def recorded_choose(current_env, now, candidates):
            choice = original_choose(current_env, now, candidates)
            decisions.append({
                "step": int(now), "action": "SELECT_ORDERING",
                "choice": str(choice),
                "candidate_lateness": {key: int(value[2]) for key, value in candidates.items()},
            })
            return choice

        policy.choose = recorded_choose

    frames = [_snapshot(env, env._elapsed_steps)] if record else []
    delay_curve = []
    previous_states = [_state_name(agent) for agent in env.agents]
    previous_positions = [_position(agent) for agent in env.agents]
    previous_malfunctions = [
        int(agent.malfunction_handler.malfunction_down_counter) for agent in env.agents
    ]
    applied_seen = set()
    previous_replans = int(getattr(policy, "replans", 0))
    done = False

    while not done:
        now = int(env._elapsed_steps)
        actions = policy.act(env)
        current_replans = int(getattr(policy, "replans", 0))
        if record and current_replans > previous_replans:
            events.append({
                "step": now, "kind": "planner",
                "message": f"Conflict-free route plan recomputed ({current_replans})",
            })
        previous_replans = current_replans

        _, _, dones, _ = env.step(actions)
        done = bool(dones["__all__"])
        step = int(env._elapsed_steps)
        current_states = [_state_name(agent) for agent in env.agents]
        current_positions = [_position(agent) for agent in env.agents]
        current_malfunctions = [
            int(agent.malfunction_handler.malfunction_down_counter) for agent in env.agents
        ]

        if record:
            for handle, (before, after) in enumerate(zip(previous_states, current_states)):
                if "MALFUNCTION" in after and "MALFUNCTION" not in before:
                    events.append({
                        "step": step, "kind": "breakdown", "train": handle,
                        "message": f"Train {handle + 1} entered malfunction state",
                    })
                if previous_malfunctions[handle] > 0 and current_malfunctions[handle] == 0:
                    events.append({
                        "step": step, "kind": "recovery", "train": handle,
                        "message": f"Train {handle + 1} malfunction period ended",
                    })
                if before != "DONE" and after == "DONE":
                    events.append({
                        "step": step, "kind": "arrival", "train": handle,
                        "message": f"Train {handle + 1} reached its destination",
                    })
                if previous_positions[handle] is None and current_positions[handle] is not None:
                    events.append({
                        "step": step, "kind": "departure", "train": handle,
                        "message": f"Train {handle + 1} entered the network",
                    })

            for disruption, status in env.disruption_status.applied.items():
                key = (disruption.train, disruption.start_step)
                if key not in applied_seen:
                    applied_seen.add(key)
                    label = {
                        "on_map": f"Train {disruption.train + 1} broke down on the network",
                        "off_map": f"Train {disruption.train + 1} breakdown delayed its departure",
                        "skipped": f"Train {disruption.train + 1} breakdown was skipped (already arrived)",
                    }.get(status, f"Train {disruption.train + 1}: {status}")
                    events.append({
                        "step": int(disruption.start_step), "kind": "disruption",
                        "train": int(disruption.train), "message": label,
                        "duration": int(disruption.duration), "status": status,
                    })

        previous_states = current_states
        previous_positions = current_positions
        previous_malfunctions = current_malfunctions
        delay_curve.append(int(network_delay(env)))
        if record:
            frames.append(_snapshot(env, step))

    arrived = sum(agent.state == TrainState.DONE for agent in env.agents)
    final_delay = int(network_delay(env))
    deadlocked = len(find_deadlocked_trains(env))
    plan = [
        {"train": int(d.train), "start_step": int(d.start_step),
         "end_step": int(d.end_step), "duration": int(d.duration)}
        for d in env.disruption_plan
    ]
    applied = [
        {"train": int(d.train), "start_step": int(d.start_step), "status": status}
        for d, status in env.disruption_status.applied.items()
    ]
    result = {
        "frames": frames,
        "events": sorted(events, key=lambda event: (event["step"], event["kind"])),
        "decisions": decisions,
        "delay_curve": delay_curve,
        "summary": {
            "policy": policy_name, "seed": int(config.seed),
            "trains": int(config.num_trains), "steps": int(env._elapsed_steps),
            "arrived": int(arrived), "final_delay": final_delay,
            "deadlocked": int(deadlocked), "disruption_plan": plan,
            "disruptions_applied": applied,
        },
    }
    if record:
        result["geometry"] = _track_geometry(env)
        result["initial_trains"] = [_train_snapshot(agent) for agent in env.agents]
    return result


def _render_html(payload):
    """Embed the episode data in a dependency-free, offline HTML replay."""
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flatland Railways · Simulation Replay</title>
<style>
:root{color-scheme:dark;--bg:#0b1120;--panel:#121c2e;--line:#26364d;--muted:#91a4bd;--text:#e6eef8;--accent:#8b5cf6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,Segoe UI,Arial,sans-serif}
header{padding:20px 24px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:16px;align-items:center}
h1{font-size:20px;margin:0}header p{margin:3px 0 0;color:var(--muted);font-size:12px}
main{max-width:1500px;margin:auto;padding:18px;display:grid;grid-template-columns:minmax(0,1.7fr) minmax(300px,.8fr);gap:16px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:14px;min-width:0}
.panel h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#b7c7dc;margin:0 0 12px}
#network{width:100%;height:auto;display:block;background:#0c1525;border-radius:9px}
.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:12px}
button,select{background:#1d2b42;color:var(--text);border:1px solid #344760;border-radius:7px;padding:8px 11px;cursor:pointer}
button:hover{border-color:var(--accent)}input[type=range]{flex:1;min-width:120px;accent-color:var(--accent)}
#stepLabel{color:var(--muted);font-variant-numeric:tabular-nums;min-width:90px}
.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:9px;margin-bottom:14px}
.metric{background:#0c1525;border:1px solid var(--line);padding:11px;border-radius:9px}
.metric strong{font-size:22px;display:block;font-variant-numeric:tabular-nums}.metric span{color:var(--muted);font-size:11px}
#trains,#events{display:flex;flex-direction:column;gap:7px;max-height:245px;overflow:auto}
.train,.event{padding:8px 9px;background:#0c1525;border:1px solid #1f3048;border-radius:8px;font-size:12px}
.train{display:flex;justify-content:space-between;gap:10px}.muted{color:var(--muted)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px}
.event small{display:block;color:var(--muted);margin-bottom:2px}
#delayChart{width:100%;height:120px;display:block;background:#0c1525;border-radius:8px;margin-top:8px}
.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:11px;margin:9px 0}
.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
.notice{color:var(--muted);font-size:11px;margin:12px 2px 0}
@media(max-width:900px){main{grid-template-columns:1fr}header{padding:16px;align-items:flex-start;flex-direction:column}}
</style>
</head>
<body>
<header><div><h1>Flatland Railways · Simulation Replay</h1><p id="subtitle"></p></div><div class="muted" id="phase">INITIAL STATE</div></header>
<main>
<section class="panel">
<h2>Railway network</h2>
<canvas id="network" width="1000" height="720"></canvas>
<div class="legend"><span><i style="background:#38bdf8"></i>Moving train (colour identifies train)</span><span><i style="background:#fb5555"></i>Malfunction</span><span><i style="background:#38bdf8;border:2px solid #e6eef8"></i>Destination · same train colour</span><span>Hollow marker = waiting at start</span></div>
<div class="controls">
<button id="play">▶ Play</button><button id="prev">◀ Step</button><button id="next">Step ▶</button>
<input id="timeline" type="range" min="0" max="0" value="0">
<span id="stepLabel">Step 0</span>
<select id="speed"><option value="1000">0.5×</option><option value="500" selected>1×</option><option value="200">2.5×</option><option value="80">5×</option></select>
</div>
<h2 style="margin-top:18px">Network delay over time</h2><canvas id="delayChart" width="1000" height="150"></canvas>
<p class="notice">This is a replay of recorded simulator states, not a separately generated animation. Playback speed does not affect simulation results.</p>
</section>
<aside style="display:flex;flex-direction:column;gap:16px">
<section class="panel"><h2>Live simulation status</h2><div class="metrics">
<div class="metric"><strong id="arrived">—</strong><span>Arrived at this step</span></div>
<div class="metric"><strong id="delay">—</strong><span>Network delay now</span></div>
<div class="metric"><strong id="onTrack">—</strong><span>Trains on track</span></div>
<div class="metric"><strong id="malfunctioning">—</strong><span>Malfunctioning now</span></div>
</div><h2 style="margin-top:14px">Final episode results</h2>
<div class="muted" id="finalSummary"></div>
<div class="muted" id="disruptionSummary" style="margin-top:8px"></div></section>
<section class="panel"><h2>Train status</h2><div id="trains"></div></section>
<section class="panel"><h2>Simulation & policy events</h2><div id="events"></div></section>
<section class="panel"><h2>Policy decisions</h2><div class="muted" id="decisionSummary"></div><div id="decisions" style="margin-top:8px"></div></section>
</aside>
</main>
<script>
const DATA = __DATA__;
const COLORS = ["#38bdf8","#34d399","#fbbf24","#c084fc","#fb7185","#2dd4bf","#f97316","#a3e635","#60a5fa","#e879f9"];
const frames = DATA.frames, geometry = DATA.geometry, summary = DATA.summary;
const $ = id => document.getElementById(id);
let index = 0, timer = null;
$("subtitle").textContent = summary.policy.toUpperCase()+" · "+summary.trains+" trains · seed "+summary.seed+" · "+summary.steps+" simulation steps";
$("finalSummary").textContent =
  summary.arrived+"/"+summary.trains+" arrived · final delay "+summary.final_delay+
  " · "+summary.deadlocked+" deadlocked · recovery "+
  (DATA.recovery_steps === null ? "not reached" : (DATA.recovery_steps === undefined ? "unavailable" : DATA.recovery_steps+" steps"));
$("disruptionSummary").textContent = summary.disruption_plan.length
  ? summary.disruption_plan.map(d=>"Train "+(d.train+1)+" · breakdown at step "+d.start_step+" · "+d.duration+" steps").join("\n")
  : "No disruptions in this scenario";
$("timeline").max = Math.max(0,frames.length-1);
function point(pos, ox, oy, cell){return [ox+(pos[1]+.5)*cell,oy+(pos[0]+.5)*cell]}
function drawNetwork(frame){
  const canvas=$("network"),ctx=canvas.getContext("2d"),W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);ctx.fillStyle="#0c1525";ctx.fillRect(0,0,W,H);
  const cell=Math.min((W-70)/geometry.width,(H-70)/geometry.height);
  const ox=(W-cell*geometry.width)/2,oy=(H-cell*geometry.height)/2;
  ctx.strokeStyle="#34445b";ctx.lineWidth=Math.max(3,cell*.13);ctx.lineCap="round";
  for(const edge of geometry.edges){const a=point(edge[0],ox,oy,cell),b=point(edge[1],ox,oy,cell);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
  if(!geometry.edges.length){ctx.fillStyle="#64748b";for(const p of geometry.cells){const q=point(p,ox,oy,cell);ctx.beginPath();ctx.arc(q[0],q[1],Math.max(2,cell*.12),0,Math.PI*2);ctx.fill()}}
  for(const train of DATA.initial_trains){
    const color=COLORS[train.id%COLORS.length];
    if(train.target){
      const q=point(train.target,ox,oy,cell);
      ctx.strokeStyle=color;ctx.lineWidth=Math.max(3,cell*.12);
      ctx.beginPath();ctx.arc(q[0],q[1],Math.max(8,cell*.27),0,Math.PI*2);ctx.stroke();
      ctx.fillStyle=color;ctx.font="bold "+Math.max(9,cell*.22)+"px Arial";
      ctx.textAlign="left";ctx.textBaseline="middle";
      ctx.fillText("D"+(train.id+1),q[0]+cell*.32,q[1]);
    }
  }
  for(const train of frame.trains){
    const color=COLORS[train.id%COLORS.length];
    const broken=train.malfunction>0||train.state.includes("MALFUNCTION");
    if(!train.position){
      if(train.start && train.state!=="DONE"){
        const q=point(train.start,ox,oy,cell);
        ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();
        ctx.arc(q[0],q[1],Math.max(8,cell*.29),0,Math.PI*2);ctx.stroke();
        ctx.fillStyle="#0c1525";ctx.beginPath();
        ctx.arc(q[0],q[1],Math.max(5,cell*.19),0,Math.PI*2);ctx.fill();
        ctx.fillStyle=color;ctx.font="bold "+Math.max(9,cell*.2)+"px Arial";
        ctx.textAlign="left";ctx.textBaseline="middle";
        ctx.fillText("T"+(train.id+1),q[0]+cell*.34,q[1]);
      }
      continue;
    }
    const q=point(train.position,ox,oy,cell);
    if(broken){ctx.strokeStyle="#fb5555";ctx.lineWidth=5;ctx.beginPath();ctx.arc(q[0],q[1],cell*.42,0,Math.PI*2);ctx.stroke()}
    ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],Math.max(8,cell*.29),0,Math.PI*2);ctx.fill();
    ctx.fillStyle="#08111f";ctx.font="bold "+Math.max(9,cell*.24)+"px Arial";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText("T"+(train.id+1),q[0],q[1]);
    if(broken){ctx.fillStyle="#fecaca";ctx.font="bold 13px Arial";ctx.fillText("!",q[0]+cell*.35,q[1]-cell*.35)}
  }
}
function drawDelay(frameIndex){
  const canvas=$("delayChart"),ctx=canvas.getContext("2d"),W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);ctx.fillStyle="#0c1525";ctx.fillRect(0,0,W,H);
  const values=DATA.delay_curve,ref=DATA.reference_curve||[];
  const max=Math.max(1,...values,...ref),left=34,right=14,top=12,bottom=24;
  ctx.strokeStyle="#25364d";ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=top+(H-top-bottom)*i/4;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(W-right,y);ctx.stroke()}
  function line(arr,color){if(!arr.length)return;ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();arr.forEach((v,i)=>{const x=left+(W-left-right)*(arr.length===1?0:i/(arr.length-1)),y=H-bottom-(H-top-bottom)*v/max;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}
  line(ref,"#64748b");line(values,"#a78bfa");
  const x=left+(W-left-right)*(frameIndex/Math.max(1,values.length));
  ctx.strokeStyle="#f8fafc";ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,H-bottom);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle="#91a4bd";ctx.font="11px Arial";ctx.fillText("0",8,H-bottom+4);ctx.fillText(String(max),8,top+4);ctx.fillText("Simulation step",W-105,H-5);
}
function phaseFor(frame){
  const active=frame.trains.some(t=>t.malfunction>0||t.state.includes("MALFUNCTION"));
  if(active)return "DISRUPTION ACTIVE";
  if(frame.step===0)return "BEFORE DISRUPTION";
  const future=summary.disruption_plan.some(d=>d.start_step>frame.step);
  if(future)return "BEFORE DISRUPTION";
  const pending=summary.disruption_plan.some(d=>d.end_step>=frame.step);
  if(pending)return "DISRUPTION WINDOW";
  return frame.arrived===summary.trains?"ALL TRAINS ARRIVED":"RECOVERY / RESUMPTION";
}
function humanState(train){
  if(train.malfunction>0||train.state.includes("MALFUNCTION"))
    return "Breakdown · "+train.malfunction+" steps remaining";
  const labels={READY_TO_DEPART:"Waiting to depart",WAITING:"Waiting",ACTIVE:"Moving",DONE:"Arrived",DONE_REMOVED:"Arrived"};
  return labels[train.state]||train.state.replaceAll("_"," ").toLowerCase();
}
function render(){
  const frame=frames[index];$("timeline").value=index;$("stepLabel").textContent="Step "+frame.step+" / "+summary.steps;
  $("phase").textContent=phaseFor(frame);drawNetwork(frame);drawDelay(index);
  $("arrived").textContent=frame.arrived+"/"+summary.trains;
  $("delay").textContent=frame.delay;
  $("onTrack").textContent=frame.on_track+"/"+summary.trains;
  $("malfunctioning").textContent=frame.malfunctioning;
  $("trains").innerHTML=frame.trains.map(t=>{
    const color=COLORS[t.id%COLORS.length],pos=t.position?"("+t.position.join(", ")+")":"At start · not on track";
    return '<div class="train"><span><i class="dot" style="background:'+color+'"></i>Train '+(t.id+1)+'<br><span class="muted">'+pos+'</span></span><b>'+humanState(t)+'</b></div>';
  }).join("");
  const visible=DATA.events.filter(e=>e.step<=frame.step).slice(-14).reverse();
  $("events").innerHTML=visible.length?visible.map(e=>'<div class="event"><small>Step '+e.step+' · '+e.kind.toUpperCase()+'</small>'+e.message+'</div>').join(""):'<div class="muted">No events recorded yet.</div>';
  const decisions=DATA.decisions.filter(d=>d.step<=frame.step);
  // Policy actions are recorded immediately before env.step(), so the state
  // shown at frame N follows decisions made at simulation step N-1.
  const decisionStep=frame.step-1;
  const atStep=decisionStep<0?[]:decisions.filter(d=>d.step===decisionStep);
  const waits=atStep.filter(d=>d.action==="WAIT").length;
  const dispatched=atStep.filter(d=>d.action==="DISPATCH").length;
  const ordering=atStep.filter(d=>d.action==="SELECT_ORDERING");
  $("decisionSummary").textContent=atStep.length
    ? ("Step "+decisionStep+": "+waits+" waiting · "+dispatched+" dispatched"+(ordering.length?" · ordering: "+ordering.map(d=>d.choice).join(", "):""))
    : "No policy decision recorded for the preceding step.";
  const recent=decisions.slice(-14).reverse();
  $("decisions").innerHTML=recent.length?recent.map(d=>{
    const label=d.action==="SELECT_ORDERING"?"Selected ordering: "+d.choice:"Train "+(d.train+1)+" → "+(d.action==="WAIT"?"Wait":"Dispatch");
    return '<div class="event"><small>Step '+d.step+'</small>'+label+'</div>';
  }).join(""):'<div class="muted">No policy decisions recorded yet.</div>';
}
function stop(){if(timer){clearInterval(timer);timer=null}$("play").textContent="▶ Play"}
function play(){if(timer){stop();return}$("play").textContent="Ⅱ Pause";timer=setInterval(()=>{if(index>=frames.length-1){stop();return}index++;render()},Number($("speed").value))}
$("play").addEventListener("click",play);
$("prev").addEventListener("click",()=>{stop();index=Math.max(0,index-1);render()});
$("next").addEventListener("click",()=>{stop();index=Math.min(frames.length-1,index+1);render()});
$("timeline").addEventListener("input",e=>{stop();index=Number(e.target.value);render()});
$("speed").addEventListener("change",()=>{if(timer){stop();play()}});
render();
</script>
</body></html>'''
    return template.replace("__DATA__", encoded)


def _render_compare_html(payload):
    """Side-by-side replay of several approaches on the same scenario, with
    one shared timeline, live metrics per approach, one combined delay chart
    and a final results table."""
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flatland Railways · Approach Comparison</title>
<style>
:root{color-scheme:dark;--bg:#0b1120;--panel:#121c2e;--line:#26364d;--muted:#91a4bd;--text:#e6eef8;--accent:#8b5cf6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,Segoe UI,Arial,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:16px;align-items:center}
h1{font-size:20px;margin:0}header p{margin:3px 0 0;color:var(--muted);font-size:12px}
main{max-width:1600px;margin:auto;padding:16px;display:flex;flex-direction:column;gap:14px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:14px;min-width:0}
.panel h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#b7c7dc;margin:0 0 10px}
.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
button,select{background:#1d2b42;color:var(--text);border:1px solid #344760;border-radius:7px;padding:8px 11px;cursor:pointer}
button:hover{border-color:var(--accent)}input[type=range]{flex:1;min-width:120px;accent-color:var(--accent)}
#stepLabel{color:var(--muted);font-variant-numeric:tabular-nums;min-width:90px}
#phase{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:12px;min-width:0}
.card h3{margin:0;font-size:16px;display:flex;align-items:center;gap:8px}.card h3 i{width:10px;height:10px;border-radius:50%;background:var(--c);display:inline-block}
.card p{margin:2px 0 8px;color:var(--muted);font-size:12px;min-height:17px}
canvas.net{width:100%;height:auto;display:block;background:#0c1525;border-radius:9px}
.live{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}
.live div{background:#0c1525;border:1px solid var(--line);border-radius:8px;padding:7px}
.live strong{display:block;font-size:18px;font-variant-numeric:tabular-nums}.live span{color:var(--muted);font-size:11px}
.status{color:var(--muted);font-size:12px;margin-top:6px;min-height:17px}
#delayChart{width:100%;height:auto;display:block;background:#0c1525;border-radius:8px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--text);font-size:12px;margin:0 0 8px}
.legend span{display:flex;align-items:center;gap:6px}.legend i{display:inline-block;width:18px;height:3px;border-radius:2px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:12px}tr.best td{font-weight:700}
.notice{color:var(--muted);font-size:11px;margin:8px 2px 0}
@media(max-width:1000px){.grid{grid-template-columns:1fr}.live{grid-template-columns:repeat(2,1fr)}header{flex-direction:column;align-items:flex-start}}
</style>
</head>
<body>
<header><div><h1>Flatland Railways · Approach Comparison</h1><p id="subtitle"></p></div><div id="phase"></div></header>
<main>
<section class="panel"><div class="controls">
<button id="play">▶ Play</button><button id="prev">◀ Step</button><button id="next">Step ▶</button>
<input id="timeline" type="range" min="0" max="0" value="0"><span id="stepLabel">Step 0</span>
<select id="speed"><option value="1000">0.5×</option><option value="500" selected>1×</option><option value="200">2.5×</option><option value="80">5×</option></select>
</div><p class="notice" id="breakdowns"></p></section>
<section class="grid" id="cards"></section>
<section class="panel"><h2>Network delay over time</h2><div class="legend" id="legend"></div>
<canvas id="delayChart" width="1500" height="260"></canvas>
<p class="notice">Network delay: how late all trains are (or will be at best) at that step. Lower is better. The dashed line is the current step; the shaded band is the breakdown.</p></section>
<section class="panel"><h2>Final results (same scenario for every approach)</h2><table id="results"></table>
<p class="notice">Extra delay from breakdowns = total delay of this run minus the same approach on the same scenario without any breakdown. This is a replay of recorded simulator states; playback speed does not change any result.</p></section>
</main>
<script>
const DATA = __DATA__;
const TRAIN_COLOURS = ["#38bdf8","#34d399","#fbbf24","#c084fc","#fb7185","#2dd4bf","#f97316","#a3e635","#60a5fa","#e879f9"];
const runs = DATA.runs, geometry = DATA.geometry, plan = DATA.disruption_plan;
const $ = id => document.getElementById(id);
const last = Math.max(...runs.map(r => r.frames.length - 1));
let index = 0, timer = null;
$("subtitle").textContent = DATA.trains+" trains · scenario seed "+DATA.seed+" · "+
  (plan.length ? plan.length+" breakdown"+(plan.length>1?"s":"")+" of "+plan.map(d=>d.duration).join(", ")+" steps" : "no breakdowns")+
  " · every approach faces the same network, timetable and breakdowns";
$("breakdowns").textContent = plan.length
  ? plan.map(d=>"Breakdown: Train "+(d.train+1)+" stops at step "+d.start_step+" for "+d.duration+" steps").join(" · ")
  : "No breakdowns in this scenario.";
$("cards").style.setProperty("--cols", runs.length > 2 ? 2 : runs.length);
$("cards").innerHTML = runs.map((r,i) =>
  '<article class="card" style="--c:'+r.colour+'"><h3><i></i>'+r.name+'</h3><p>'+r.description+'</p>'+
  '<canvas class="net" id="net'+i+'" width="900" height="620"></canvas>'+
  '<div class="live"><div><strong id="arr'+i+'">-</strong><span>Arrived</span></div>'+
  '<div><strong id="del'+i+'">-</strong><span>Delay now</span></div>'+
  '<div><strong id="on'+i+'">-</strong><span>On track</span></div>'+
  '<div><strong id="brk'+i+'">-</strong><span>Broken down</span></div></div>'+
  '<div class="status" id="st'+i+'"></div></article>').join("");
$("legend").innerHTML = runs.map(r => '<span><i style="background:'+r.colour+'"></i>'+r.name+'</span>').join("");
$("timeline").max = last;
function point(pos, ox, oy, cell){return [ox+(pos[1]+.5)*cell,oy+(pos[0]+.5)*cell]}
// Zoom to the part of the grid that has track, so the network fills the panel.
const used=geometry.cells.length?geometry.cells:[[0,0],[geometry.height-1,geometry.width-1]];
const box={r0:Math.min(...used.map(c=>c[0])),r1:Math.max(...used.map(c=>c[0])),
           c0:Math.min(...used.map(c=>c[1])),c1:Math.max(...used.map(c=>c[1]))};
function drawNetwork(canvas, frame){
  const ctx=canvas.getContext("2d"),W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);ctx.fillStyle="#0c1525";ctx.fillRect(0,0,W,H);
  const rows=box.r1-box.r0+1,cols=box.c1-box.c0+1;
  const cell=Math.min((W-40)/cols,(H-40)/rows);
  const ox=(W-cell*cols)/2-box.c0*cell,oy=(H-cell*rows)/2-box.r0*cell;
  ctx.strokeStyle="#34445b";ctx.lineWidth=Math.max(3,cell*.13);ctx.lineCap="round";
  for(const e of geometry.edges){const a=point(e[0],ox,oy,cell),b=point(e[1],ox,oy,cell);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
  for(const t of DATA.initial_trains){
    if(!t.target)continue;const c=TRAIN_COLOURS[t.id%TRAIN_COLOURS.length],q=point(t.target,ox,oy,cell);
    ctx.strokeStyle=c;ctx.lineWidth=Math.max(2,cell*.1);ctx.beginPath();ctx.arc(q[0],q[1],Math.max(7,cell*.27),0,Math.PI*2);ctx.stroke();
  }
  for(const t of frame.trains){
    const c=TRAIN_COLOURS[t.id%TRAIN_COLOURS.length],broken=t.malfunction>0||t.state.includes("MALFUNCTION");
    if(!t.position){
      if(t.start&&t.state!=="DONE"){const q=point(t.start,ox,oy,cell);ctx.strokeStyle=c;ctx.lineWidth=3;ctx.beginPath();ctx.arc(q[0],q[1],Math.max(7,cell*.29),0,Math.PI*2);ctx.stroke()}
      continue;
    }
    const q=point(t.position,ox,oy,cell);
    if(broken){ctx.strokeStyle="#fb5555";ctx.lineWidth=5;ctx.beginPath();ctx.arc(q[0],q[1],cell*.45,0,Math.PI*2);ctx.stroke()}
    ctx.fillStyle=c;ctx.beginPath();ctx.arc(q[0],q[1],Math.max(7,cell*.31),0,Math.PI*2);ctx.fill();
    ctx.fillStyle="#08111f";ctx.font="bold "+Math.max(8,cell*.26)+"px Arial";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(String(t.id+1),q[0],q[1]);
  }
}
function drawDelay(){
  const canvas=$("delayChart"),ctx=canvas.getContext("2d"),W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);ctx.fillStyle="#0c1525";ctx.fillRect(0,0,W,H);
  const left=50,right=16,top=14,bottom=30,steps=Math.max(1,last);
  const max=Math.max(1,...runs.flatMap(r=>r.delay_curve));
  const X=s=>left+(W-left-right)*s/steps, Y=v=>H-bottom-(H-top-bottom)*v/max;
  for(const d of plan){ctx.fillStyle="rgba(251,85,85,.12)";ctx.fillRect(X(d.start_step),top,X(d.end_step+1)-X(d.start_step),H-top-bottom)}
  ctx.strokeStyle="#25364d";ctx.lineWidth=1;ctx.fillStyle="#91a4bd";ctx.font="12px Arial";ctx.textAlign="right";ctx.textBaseline="middle";
  for(let i=0;i<=4;i++){const v=Math.round(max*i/4),y=Y(v);ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(W-right,y);ctx.stroke();ctx.fillText(String(v),left-8,y)}
  ctx.textAlign="center";ctx.textBaseline="top";
  for(let i=0;i<=5;i++){const s=Math.round(steps*i/5);ctx.fillText(String(s),X(s),H-bottom+8)}
  ctx.textAlign="right";ctx.textBaseline="bottom";ctx.fillText("Simulation step",W-right-4,H-bottom-4);
  for(const r of runs){
    ctx.strokeStyle=r.colour;ctx.lineWidth=2.5;ctx.beginPath();
    r.delay_curve.forEach((v,i)=>{const x=X(i+1),y=Y(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();
    const n=r.delay_curve.length;if(n){ctx.fillStyle=r.colour;ctx.beginPath();ctx.arc(X(n),Y(r.delay_curve[n-1]),4,0,Math.PI*2);ctx.fill()}
  }
  ctx.strokeStyle="#f8fafc";ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(X(index),top);ctx.lineTo(X(index),H-bottom);ctx.stroke();ctx.setLineDash([]);
}
function resultsTable(){
  const best=Math.min(...runs.map(r=>r.summary.final_delay));
  const head='<tr><th>Approach</th><th>Arrived</th><th>Total delay (steps)</th><th>Extra delay from breakdowns</th><th>Deadlocked trains</th><th>Steps to finish</th></tr>';
  $("results").innerHTML=head+runs.map(r=>{const s=r.summary;
    return '<tr class="'+(s.final_delay===best?"best":"")+'"><td><span style="color:'+r.colour+'">●</span> '+r.name+'</td><td>'+s.arrived+'/'+s.trains+'</td><td>'+s.final_delay+'</td><td>'+(r.extra_delay===null?"n/a":r.extra_delay)+'</td><td>'+s.deadlocked+'</td><td>'+s.steps+'</td></tr>'}).join("");
}
function phaseFor(step){
  if(plan.some(d=>d.start_step<=step&&step<=d.end_step))return "BREAKDOWN ACTIVE";
  if(plan.some(d=>d.start_step>step))return "BEFORE BREAKDOWN";
  return plan.length?"AFTER BREAKDOWN":"";
}
function render(){
  $("timeline").value=index;$("stepLabel").textContent="Step "+index+" / "+last;$("phase").textContent=phaseFor(index);
  runs.forEach((r,i)=>{
    const f=r.frames[Math.min(index,r.frames.length-1)];
    drawNetwork($("net"+i),f);
    $("arr"+i).textContent=f.arrived+"/"+DATA.trains;$("del"+i).textContent=f.delay;
    $("on"+i).textContent=f.on_track;$("brk"+i).textContent=f.malfunctioning;
    const done=index>=r.frames.length-1;
    const ord=r.decisions.filter(d=>d.action==="SELECT_ORDERING"&&d.step<index).slice(-1)[0];
    const waits=r.decisions.filter(d=>d.step===index-1&&d.action==="WAIT").length;
    $("st"+i).textContent=done?("Finished at step "+r.summary.steps+": total delay "+r.summary.final_delay)
      :(ord?"Ordering rule in use: "+ord.choice.toUpperCase():(r.key==="mappo"?(waits?waits+" train(s) told to wait this step":"No train held back this step"):""));
  });
  drawDelay();
}
function stop(){if(timer){clearInterval(timer);timer=null}$("play").textContent="▶ Play"}
function play(){if(timer){stop();return}$("play").textContent="Ⅱ Pause";timer=setInterval(()=>{if(index>=last){stop();return}index++;render()},Number($("speed").value))}
$("play").addEventListener("click",play);
$("prev").addEventListener("click",()=>{stop();index=Math.max(0,index-1);render()});
$("next").addEventListener("click",()=>{stop();index=Math.min(last,index+1);render()});
$("timeline").addEventListener("input",e=>{stop();index=Number(e.target.value);render()});
$("speed").addEventListener("change",()=>{if(timer){stop();play()}});
resultsTable();render();
</script>
</body></html>"""
    return template.replace("__DATA__", encoded)


def _run_with_reference(config, spec, policy_name, model, skip_reference):
    """Record one approach on the scenario, plus (optionally) the same approach
    without breakdowns, to measure the extra delay the breakdowns caused."""
    _seed_torch(config.seed)
    result = _run_episode(config, spec, policy_name, model, record=True)
    reference_curve, recovery = None, None
    if spec is not None and result["summary"]["disruption_plan"] and not skip_reference:
        reference_curve = _run_episode(config, None, policy_name, model, record=False)["delay_curve"]
        recovery = recovery_time(
            result["delay_curve"], reference_curve,
            [type("RecordedDisruption", (), d)() for d in result["summary"]["disruption_plan"]],
        )
    result["reference_curve"] = reference_curve
    result["recovery_steps"] = recovery
    return result


def _main_compare(args):
    config = EnvConfig(num_trains=args.trains, seed=args.seed)
    spec = DisruptionSpec(count=args.disruptions, duration=args.duration) if args.disruptions else None
    runs = []
    for name in args.compare:
        model = None
        if name in ("ppo", "rl_strategy"):
            model = args.ppo_model
        elif name == "mappo":
            model = args.mappo_model or _default_mappo_model()
        result = _run_with_reference(config, spec, name, model, args.skip_reference)
        summary = result["summary"]
        reference = result["reference_curve"]
        runs.append({
            "key": name,
            "name": DISPLAY_NAMES.get(name, name),
            "description": DESCRIPTIONS.get(name, ""),
            "colour": COMPARE_COLOURS.get(name, "#94a3b8"),
            "frames": result["frames"],
            "events": result["events"],
            "decisions": result["decisions"],
            "delay_curve": result["delay_curve"],
            "summary": summary,
            "extra_delay": (summary["final_delay"] - reference[-1]) if reference else None,
            "recovery_steps": result["recovery_steps"],
        })
        print(f"{DISPLAY_NAMES.get(name, name):>22}: arrived {summary['arrived']}/{args.trains}, "
              f"total delay {summary['final_delay']}, deadlocked {summary['deadlocked']}"
              + (f", extra delay from breakdowns {runs[-1]['extra_delay']}"
                 if runs[-1]["extra_delay"] is not None else ""))
    payload = {
        "trains": args.trains, "seed": args.seed,
        "disruption_plan": runs[0]["summary"]["disruption_plan"],
        "geometry": _recorded_geometry(config, spec),
        "initial_trains": None,
        "runs": runs,
    }
    payload["initial_trains"] = payload["geometry"].pop("initial_trains")
    output = Path(args.output or f"outputs/compare_seed{args.seed}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_compare_html(payload), encoding="utf-8")
    print(f"Saved comparison replay: {output}")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())


def _recorded_geometry(config, spec):
    """Track layout and train start/target markers. The scenario seed fixes
    them, so they are the same for every approach."""
    env, _, _ = make_env(config, spec)
    geometry = _track_geometry(env)
    geometry["initial_trains"] = [_train_snapshot(agent) for agent in env.agents]
    return geometry


def main():
    parser = argparse.ArgumentParser(description="Record and replay a Flatland railway episode in your browser.")
    names = ["mappo", "ppo", "rl_strategy", "fcfs", "priority", "edf", "spf",
             "greedy_select", "release_all", "hold_if_blocked"]
    parser.add_argument("--policy", choices=names, default="mappo")
    parser.add_argument("--model", help="MAPPO .pt checkpoint or PPO strategy .zip checkpoint")
    parser.add_argument("--compare", nargs="+", choices=names, metavar="POLICY",
                        help="run 2 to 4 approaches on the same scenario and show them side by side, "
                             "e.g. --compare fcfs priority ppo mappo")
    parser.add_argument("--ppo-model", default=DEFAULT_PPO_MODEL, help="PPO model used by --compare")
    parser.add_argument("--mappo-model", default=None,
                        help="MAPPO model used by --compare (default: results/models/mappo_seed1.pt)")
    parser.add_argument("--trains", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disruptions", type=int, default=1)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--output", default=None, help="HTML output path (default: outputs/replay_<policy>_seed<seed>.html)")
    parser.add_argument("--skip-reference", action="store_true", help="skip the undisrupted reference run used for recovery comparison")
    parser.add_argument("--no-open", action="store_true", help="write the HTML file without opening a browser")
    args = parser.parse_args()

    if args.trains < 1 or args.disruptions < 0 or args.duration < 1:
        parser.error("--trains must be positive, --disruptions non-negative, and --duration positive")
    if args.compare:
        if not 2 <= len(args.compare) <= 4 or len(set(args.compare)) != len(args.compare):
            parser.error("--compare needs 2 to 4 different policies")
        return _main_compare(args)
    if args.policy == "rl_strategy" and not args.model:
        parser.error("--model is required for --policy rl_strategy")

    config = EnvConfig(num_trains=args.trains, seed=args.seed)
    spec = DisruptionSpec(count=args.disruptions, duration=args.duration) if args.disruptions else None
    _seed_torch(args.seed)
    result = _run_episode(config, spec, args.policy, args.model, record=True)

    reference_curve = None
    recovery = None
    if spec is not None and result["summary"]["disruption_plan"] and not args.skip_reference:
        reference = _run_episode(config, None, args.policy, args.model, record=False)
        reference_curve = reference["delay_curve"]
        recovery = recovery_time(
            result["delay_curve"], reference_curve,
            # Rebuild the exact immutable disruption records expected by the metric.
            [type("RecordedDisruption", (), d)() for d in result["summary"]["disruption_plan"]],
        )

    result["reference_curve"] = reference_curve
    result["recovery_steps"] = recovery
    output = Path(args.output or f"outputs/replay_{args.policy}_seed{args.seed}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_html(result), encoding="utf-8")
    print(f"Saved simulation replay: {output}")
    print(
        f"policy={args.policy} seed={args.seed} "
        f"arrived={result['summary']['arrived']}/{args.trains} "
        f"delay={result['summary']['final_delay']} "
        f"deadlocked={result['summary']['deadlocked']}"
    )
    if recovery is not None:
        print(f"recovery_steps={recovery}")
    elif spec is not None and not args.skip_reference:
        print("recovery_steps=not reached before episode end")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
