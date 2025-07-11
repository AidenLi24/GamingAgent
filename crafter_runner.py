import os
import time
from gamingagent.envs.custom_05_crafter.crafterEnv import CrafterEnvWrapper
import numpy as np
import random


# Basic config values to hardcode for now
game_name = "crafter"
obs_mode = "vision"
cache_dir = "cache/crafter/test_run"
script_dir = os.path.dirname(os.path.abspath(__file__))
game_config_path = os.path.abspath(os.path.join(script_dir, "../gamingagent/envs/custom_05_crafter/game_env_config.json"))


# Create the adapter log directory
os.makedirs(cache_dir, exist_ok=True)
# Instantiate environment manually using values you control
env = CrafterEnvWrapper(
   game_name_for_adapter=game_name,
   observation_mode_for_adapter=obs_mode,
   agent_cache_dir_for_adapter=cache_dir,
   game_specific_config_path_for_adapter=game_config_path,
   max_stuck_steps_for_adapter=20,
)

# Run 1 test episode
episode_id = 1
obs, info = env.reset(seed=42, episode_id=episode_id)
obs_prev = None
available_actions = ["move_left", "move_right", "move_up", "move_down", "do"]
total_reward = 0  # Track cumulative reward
for step in range(200):
   chosen_action = random.choice(available_actions)
   obs, reward, terminated, truncated, info, perf_score = env.step(chosen_action)

   total_reward += reward  # Accumulate reward
   if obs_prev is not None:
      is_different = not np.array_equal(obs, obs_prev)
      print(f"Step {step}: Action={chosen_action}, Frame changed? {is_different}")
   else:
      print(f"Step {step}: Action={chosen_action}")

   obs_prev = obs
   print(f"Reward: {reward}, Terminated: {terminated}")
   if terminated or truncated:
      break
env.close()
print(f"Episode finished. Total reward: {total_reward:.2f}")




