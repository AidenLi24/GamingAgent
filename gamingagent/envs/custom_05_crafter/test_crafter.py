from gamingagent.envs.custom_05_crafter.crafterEnv import CrafterEnvWrapper

env = CrafterEnvWrapper()
obs, info = env.reset()
print("obs.shape:", obs.shape)

for i in range(20):
    a = env.action_space.sample()
    obs, rew, done, trunc, info = env.step(a)
    print(f"step {i:2d} act={a:2d} rew ={rew:+.2f} done={done}")
    if done:
        break