import gymnasium as gym
import crafter
from typing import Optional
import numpy as np
from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation
class CrafterEnvWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self,
                 render_mode: Optional[str] = None,
                 game_name_for_adapter: str = "crafter",
                 observation_mode_for_adapter: str = "vision",
                 agent_cache_dir_for_adapter: str = "cache/crafter/default_run",
                 game_specific_config_path_for_adapter: str = "gamingagent/envs/custom_05_crafter/game_env_config.json",
                 max_stuck_steps_for_adapter: Optional[int] = 20,
                 view=(9, 9),
                 size=(64, 64)):
        super().__init__()
        # Normalize `view` and `size` to tuples (merge-safe)
        if isinstance(view, np.ndarray):
            view = tuple(view.tolist())
        elif not isinstance(view, tuple):
            view = (view, view)
        if isinstance(size, np.ndarray):
            size = tuple(size.tolist())
        elif not isinstance(size, tuple):
            size = (size, size)
        # Instantiate crafter.Env with normalized types
        self.env = crafter.Env(view=view, size=size)
        self.render_mode = render_mode
        self.num_env_steps = 0
        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter
        )
        # Basic movement + interaction
        self.adapter.set_action_mapping({
            "left": 1,
            "right": 2,
            "up": 3,
            "down": 4,
            "do": 5
        })
    def reset(self, *, seed=None, options=None, episode_id=1):
        super().reset(seed=seed)
        obs = self.env.reset()
        # ✅ Ensure step counter exists (no env.py edits required)
        if not hasattr(self.env, "_step") or self.env._step is None:
            self.env._step = 0
        return obs, {}
    def step(self, agent_action_str, thought_process="", time_taken_s=0.0):
        self.adapter.increment_step()
        action_idx = self.adapter.map_agent_action_to_env_action(agent_action_str)
        obs, reward, done, info = self.env.step(action_idx)
        agent_obs = self.adapter.create_agent_observation(img_path=None, text_representation=None)
        self.adapter.log_step_data(agent_action_str, thought_process, reward, info, done, False, time_taken_s, reward, agent_obs)
        return agent_obs, reward, done, False, info, reward
    def render(self, mode=None):
        return self.env.render(mode or self.metadata["render_modes"][0])
    def close(self):
        self.env.close()
        self.adapter.close_log_file()


