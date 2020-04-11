from gym.envs.registration import register

register(
    id='CommonsGame-v0',
    entry_point='CommonsGame.envs:CommonsGame',
)