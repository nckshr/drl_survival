from gymnasium.envs.registration import register

register(
    id='survival_envs/Survival',
    entry_point='environments.envs:SurvivalEnv',
    max_episode_steps=1000,
)

register(
    id='survival_envs/SurvivalSimple',
    entry_point='environments.envs:SurvivalEnvSimple',
    max_episode_steps=1000,
)