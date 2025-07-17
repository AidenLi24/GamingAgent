import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from gamingagent.envs.custom_05_crafter.crafterEnv import CrafterEnvWrapper
# Basic config
game_name = "crafter"
obs_mode = "vision"
cache_dir = "cache/crafter/test_run"
script_dir = os.path.dirname(os.path.abspath(__file__))
game_config_path = os.path.abspath(os.path.join(script_dir, "../gamingagent/envs/custom_05_crafter/game_env_config.json"))
os.makedirs(cache_dir, exist_ok=True)
# Instantiate environment
env = CrafterEnvWrapper(
    game_name_for_adapter=game_name,
    observation_mode_for_adapter=obs_mode,
    agent_cache_dir_for_adapter=cache_dir,
    game_specific_config_path_for_adapter=game_config_path,
    max_stuck_steps_for_adapter=20,
)
# Reset environment
episode_id = 1
obs, info = env.reset(seed=42, episode_id=episode_id)
# Random actions for testing
actions = ["left", "right", "up", "down", "do"]
total_reward = 0
plt.ion()  # Interactive mode ON for live updating
for step in range(200):
    action = random.choice(actions)
    obs, reward, terminated, truncated, info, perf_score = env.step(action)
    total_reward += reward
    # Render the frame visually
    frame = env.render((64, 64))
    plt.imshow(frame)
    #plt.axis("off")
    plt.title(f"Step: {step}, Action: {action}, Reward: {reward:.2f}")
    plt.pause(0.1)
    plt.clf()
    if terminated or truncated:
        break
plt.ioff()  # Turn off interactive mode
plt.show()
env.close()
print(f"Episode finished. Total Reward: {total_reward}")



