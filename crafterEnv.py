import gymnasium as gym
import crafter
from typing import Optional, Dict, Any
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
                 max_stuck_steps_for_adapter: Optional[int] = 20):
        super().__init__()
        self.env = crafter.Env()
        self.render_mode = render_mode
        self.num_env_steps = 0
        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter
        )
        # Action mapping matches crafter.yaml’s 'actions' list
        self.adapter.set_action_mapping({
            "noop": 0,
            "move_left": 1,
            "move_right": 2,
            "move_up": 3,
            "move_down": 4,
            "do": 5,
            "sleep": 6,
            "place_stone": 7,
            "place_table": 8,
            "place_furnace": 9,
            "place_plant": 10,
            "make_wood_pickaxe": 11,
            "make_stone_pickaxe": 12,
            "make_iron_pickaxe": 13,
            "make_wood_sword": 14,
            "make_stone_sword": 15,
            "make_iron_sword": 16
        })
    def reset(self, *, seed=None, options=None, episode_id=1):
        super().reset(seed=seed)
        obs = self.env.reset()
        if not hasattr(self.env, "_step") or self.env._step is None:
            self.env._step = 0
        return obs, {}
    def step(self, agent_action_str: str, thought_process: str = "", time_taken_s: float = 0.0):
        self.adapter.increment_step()
        action_idx = self.adapter.map_agent_action_to_env_action(agent_action_str)
        obs, reward, done, info = self.env.step(action_idx)
        agent_obs = self.adapter.create_agent_observation(img_path=None, text_representation=None)
        self.adapter.log_step_data(
            agent_action_str, thought_process,
            reward, info, done, False, time_taken_s,
            reward, agent_obs
        )
        return agent_obs, reward, done, False, info, reward
    def render(self, mode=None):
        # Try to render using preferred mode, fallback if unavailable
        mode = mode or self.metadata["render_modes"][0]
        try:
            return self.env.render(mode)
        except Exception as e:
            print(f"[Render Warning] Failed to render: {e}")
            return None
    def close(self):
        self.env.close()
        self.adapter.close_log_file()


