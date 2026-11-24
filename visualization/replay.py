"""Create a self-contained, replayable HTML visualization of a Flatland episode.

No visualization dependency is required: the renderer is plain HTML Canvas.
All positions, states, disruption events and policy decisions are recorded from
the live Flatland environment; the browser only replays those recorded facts.
"""
from __future__ import annotations

import argparse
import json
import os
import webbrowser
from pathlib import Path

import numpy as np
from flatland.envs.step_utils.states import TrainState

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import EnvConfig, make_env
from railsim.metrics import network_delay, recovery_time
from railsim.train_status import find_deadlocked_trains


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
TRAIN_COLORS = ["#38bdf8", "#34d399", "#fbbf24", "#c084fc", "#fb7185",
                "#2dd4bf", "#f97316", "#a3e635", "#60a5fa", "#e879f9"]


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
                transitions = rail.get_transitions(row, col, incoming)
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
    start = [int(value) for value in config[0]] if config and config[0] is not None else None
    target = [int(value) for value in agent.target] if agent.target is not None else None
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
        MAPPODispatcher.model_path = model or "models/mappo_seed1.pt"
        MAPPODispatcher._actor = None
        MAPPODispatcher._loaded_from = None
        return MAPPODispatcher()
    if name == "rl_strategy":
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
                        "message": f"Train {handle} entered malfunction state",
                    })
                if previous_malfunctions[handle] > 0 and current_malfunctions[handle] == 0:
                    events.append({
                        "step": step, "kind": "recovery", "train": handle,
                        "message": f"Train {handle} malfunction period ended",
                    })
                if before != "DONE" and after == "DONE":
                    events.append({
                        "step": step, "kind": "arrival", "train": handle,
                        "message": f"Train {handle} reached its target",
                    })
                if previous_positions[handle] is None and current_positions[handle] is not None:
                    events.append({
                        "step": step, "kind": "departure", "train": handle,
                        "message": f"Train {handle} entered the network",
                    })

            for disruption, status in env.disruption_status.applied.items():
                key = (disruption.train, disruption.start_step)
                if key not in applied_seen:
                    applied_seen.add(key)
                    label = {
                        "on_map": f"Train {disruption.train} broke down on the network",
                        "off_map": f"Train {disruption.train} breakdown delayed its departure",
                        "skipped": f"Train {disruption.train} breakdown was skipped (already arrived)",
                    }.get(status, f"Train {disruption.train}: {status}")
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
<div class="legend"><span><i style="background:#38bdf8"></i>Train</span><span><i style="background:#fb5555"></i>Malfunction</span><span><i style="background:#fbbf24"></i>Target</span><span>Track layout from Flatland transition map</span></div>
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
<section class="panel"><h2>Episode metrics</h2><div class="metrics">
<div class="metric"><strong id="arrived">—</strong><span>Trains arrived</span></div>
<div class="metric"><strong id="delay">—</strong><span>Final network delay</span></div>
<div class="metric"><strong id="deadlocked">—</strong><span>Deadlocked trains</span></div>
<div class="metric"><strong id="recovery">—</strong><span>Recovery time (steps)</span></div>
</div><div class="muted" id="disruptionSummary"></div></section>
<section class="panel"><h2>Train status</h2><div id="trains"></div></section>
<section class="panel"><h2>Simulation & policy events</h2><div id="events"></div></section>
<section class="panel"><h2>Policy decisions</h2><div id="decisions"></div></section>
</aside>
</main>
<script>
const DATA = __DATA__;
const COLORS = ["#38bdf8","#34d399","#fbbf24","#c084fc","#fb7185","#2dd4bf","#f97316","#a3e635","#60a5fa","#e879f9"];
const frames = DATA.frames, geometry = DATA.geometry, summary = DATA.summary;
const $ = id => document.getElementById(id);
let index = 0, timer = null;
$("subtitle").textContent = summary.policy.toUpperCase()+" · "+summary.trains+" trains · seed "+summary.seed+" · "+summary.steps+" simulation steps";
$("arrived").textContent = summary.arrived+"/"+summary.trains;
$("delay").textContent = summary.final_delay;
$("deadlocked").textContent = summary.deadlocked;
$("recovery").textContent = DATA.recovery_steps === null ? "Not reached" : (DATA.recovery_steps === undefined ? "—" : DATA.recovery_steps);
$("disruptionSummary").textContent = summary.disruption_plan.length
  ? summary.disruption_plan.map(d=>"Train "+d.train+" · step "+d.start_step+" · "+d.duration+" steps").join("\n")
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
    if(train.target){const q=point(train.target,ox,oy,cell);ctx.fillStyle="#fbbf24";ctx.globalAlpha=.85;ctx.fillRect(q[0]-cell*.18,q[1]-cell*.18,cell*.36,cell*.36);ctx.globalAlpha=1}
  }
  for(const train of frame.trains){
    if(!train.position)continue;
    const q=point(train.position,ox,oy,cell),color=COLORS[train.id%COLORS.length];
    const broken=train.malfunction>0||train.state.includes("MALFUNCTION");
    if(broken){ctx.strokeStyle="#fb5555";ctx.lineWidth=5;ctx.beginPath();ctx.arc(q[0],q[1],cell*.42,0,Math.PI*2);ctx.stroke()}
    ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],Math.max(8,cell*.29),0,Math.PI*2);ctx.fill();
    ctx.fillStyle="#08111f";ctx.font="bold "+Math.max(9,cell*.24)+"px Arial";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText("T"+train.id,q[0],q[1]);
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
function render(){
  const frame=frames[index];$("timeline").value=index;$("stepLabel").textContent="Step "+frame.step+" / "+summary.steps;
  $("phase").textContent=phaseFor(frame);drawNetwork(frame);drawDelay(index);
  $("trains").innerHTML=frame.trains.map(t=>{
    const color=COLORS[t.id%COLORS.length],pos=t.position?"("+t.position.join(", ")+")":"Off map";
    const status=t.malfunction>0?"MALFUNCTION · "+t.malfunction+" left":t.state;
    return '<div class="train"><span><i class="dot" style="background:'+color+'"></i>Train '+t.id+'<br><span class="muted">'+pos+'</span></span><b>'+status+'</b></div>';
  }).join("");
  const visible=DATA.events.filter(e=>e.step<=frame.step).slice(-14).reverse();
  $("events").innerHTML=visible.length?visible.map(e=>'<div class="event"><small>Step '+e.step+' · '+e.kind.toUpperCase()+'</small>'+e.message+'</div>').join(""):'<div class="muted">No events recorded yet.</div>';
  const decisions=DATA.decisions.filter(d=>d.step<=frame.step).slice(-14).reverse();
  $("decisions").innerHTML=decisions.length?decisions.map(d=>{
    const label=d.action==="SELECT_ORDERING"?"Selected ordering: "+d.choice:"Train "+d.train+" → "+d.action;
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


def main():
    parser = argparse.ArgumentParser(description="Record and replay a Flatland railway episode in your browser.")
    parser.add_argument("--policy", choices=[
        "mappo", "rl_strategy", "fcfs", "priority", "edf", "spf",
        "greedy_select", "release_all", "hold_if_blocked",
    ], default="mappo")
    parser.add_argument("--model", help="MAPPO .pt checkpoint or PPO strategy .zip checkpoint")
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
    if args.policy == "rl_strategy" and not args.model:
        parser.error("--model is required for --policy rl_strategy")

    config = EnvConfig(num_trains=args.trains, seed=args.seed)
    spec = DisruptionSpec(count=args.disruptions, duration=args.duration) if args.disruptions else None
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
