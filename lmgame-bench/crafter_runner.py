import os
import random
import time

import numpy as np
from gamingagent.envs.custom_05_crafter.crafterEnv import CrafterEnvWrapper
import pygame
from PIL import Image

# -------------------------------
# Basic config values
# -------------------------------
game_name = "crafter"
obs_mode = "vision"
cache_dir = "cache/crafter/test_run"
script_dir = os.path.dirname(os.path.abspath(__file__))
game_config_path = os.path.abspath(
    os.path.join(script_dir, "../gamingagent/envs/custom_05_crafter/game_env_config.json")
)

# Create the adapter log directory
os.makedirs(cache_dir, exist_ok=True)

# Instantiate environment manually
env = CrafterEnvWrapper(
    game_name_for_adapter=game_name,
    observation_mode_for_adapter=obs_mode,
    agent_cache_dir_for_adapter=cache_dir,
    game_specific_config_path_for_adapter=game_config_path,
    max_stuck_steps_for_adapter=20,
)

# -------------------------------
# Pygame Setup
# -------------------------------
pygame.init()
window_size = (512, 512)  # Resize to something comfortable
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Crafter Pygame Rendering")
env.env._size = np.array((512,512), dtype=np.int32)

# -------------------------------
# Run test episode
# -------------------------------
episode_id = 1
obs, info = env.reset(seed=42, episode_id=episode_id)

total_reward = 0

for step in range(250):
    # Choose random action (cardinal + "do")
    action = random.choice(["left", "right", "up", "down", "do"])
    obs, reward, terminated, truncated, info, perf_score = env.step(action)
    total_reward += reward

    # ✅ Directly fetch raw RGB from underlying Crafter env
    raw_frame = env.env.render()

    # Convert to Pygame surface
    img = Image.fromarray(raw_frame, "RGB").resize(window_size)
    img_surface = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
    screen.blit(img_surface, (0, 0))
    pygame.display.flip()
    time.sleep(0.05)

    # Basic event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    if terminated or truncated:
        break

pygame.quit()
env.close()
print(f"Episode finished. Total reward: {total_reward}")

