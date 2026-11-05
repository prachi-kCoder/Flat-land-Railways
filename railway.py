# ============================================================
# Flatland-RL: Railway Network Visualisation
# ============================================================
#
# Builds the base scenario and draws the railway network.
#
# Run:
#   python railway.py                      # open a window
#   python railway.py --save network.png   # save an image instead
#                                          # (works without a screen)
# ============================================================

import argparse

from PIL import Image
from flatland.utils.rendertools import RenderTool

from railsim import EnvConfig, direction_name, make_env, train_target


parser = argparse.ArgumentParser(description="Draw the Flatland railway network.")
parser.add_argument("--save", metavar="FILE", help="save the drawing to an image file instead of opening a window")
args = parser.parse_args()


CONFIG = EnvConfig(width=30, height=30, num_trains=3, seed=42)

env, obs, info = make_env(CONFIG)


print("=" * 65)
print("        FLATLAND-RL INITIAL RAILWAY SIMULATION")
print("=" * 65)
print(f"Grid Size        : {CONFIG.width} x {CONFIG.height}")
print(f"Number of Trains : {CONFIG.num_trains}")
print(f"Random Seed      : {CONFIG.seed}")


print("\nTRAIN INFORMATION")
print("-" * 65)

for agent in env.agents:
    start_cell, start_direction = agent.initial_configuration
    print(
        f"Train {agent.handle}: "
        f"Start={start_cell} facing {direction_name(start_direction)}, "
        f"Target={train_target(agent)}"
    )


print("\nRendering railway environment...")

renderer = RenderTool(env)

if args.save:
    # Note: at step 0 no train has departed yet, so the image shows
    # the network and stations only.
    renderer.render_env(show=False, show_observations=False, show_predictions=False)
    Image.fromarray(renderer.get_image()).save(args.save)
    print(f"Saved railway drawing to {args.save}")
else:
    renderer.render_env(show=True, show_observations=False, show_predictions=False)
    input("\nPress ENTER to close the simulation...")
    renderer.close_window()

print("Initial Flatland-RL environment setup completed.")
