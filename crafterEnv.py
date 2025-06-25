import gymnasium as gym
import crafter

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation

class CrafterEnvWrapper(gym.Env):
    metadata = {"render_modes": ["human","rgb_array"], "render_fps": 30}

    def __init__(self,
                 area=(64,64), view=(9,9), size=(64,64), length=10000, seed=None,
                 game_name_for_adapter="crafter",
                 observation_mode_for_adapter="vision",
                 agent_cache_dir_for_adapter="cache/crafter/default_run",
                 game_specific_config_path_for_adapter="gamingagent/envs/custom_05_crafter/game_env_config.json",
                 max_stuck_steps_for_adapter=20):
        super().__init__()
        # 1) Instantiate the real Crafter environment:
        self.env = crafter.Env(area=area, view=view, size=size, length=length, seed=seed)

        # 2) Pull in its action and observation spaces:
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # 3) Hook up the adapter for logging & formatting:
        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter
        )

    def reset(self, *, seed=None, options=None, episode_id=1):
        # reset both adapter and underlying env
        self.adapter.reset_episode(episode_id)
        obs = self.env.reset(seed=seed)
        # wrap obs (numpy array) into an Observation via adapter…
        agent_obs = self.adapter.create_agent_observation(img_path=None, text_representation=None)
        return agent_obs, {}

    def step(self, agent_action_str, thought_process="", time_taken_s=0.0):
        self.adapter.increment_step()
        # map the LLM’s string to an integer action
        action_idx = self.adapter.map_agent_action_to_env_action(agent_action_str)
        obs, reward, done, info = self.env.step(action_idx)
        # wrap and log …
        agent_obs = self.adapter.create_agent_observation(img_path=None, text_representation=None)
        self.adapter.log_step_data(agent_action_str, thought_process, reward, info, done, False, time_taken_s, reward, agent_obs)
        return agent_obs, reward, done, False, info, reward

    def render(self, mode=None):
        return self.env.render(mode or self.metadata["render_modes"][0])

    def close(self):
        self.env.close()
        self.adapter.close_log_file()