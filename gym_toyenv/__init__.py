from gym.envs.registration import register

register(
    id='toyenv-v0',
    entry_point='gym_toyenv.envs:ToyEnv',
    max_episode_steps=200
)