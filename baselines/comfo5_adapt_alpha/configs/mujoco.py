import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "halfcheetah-medium-expert-v2"
    info = "20230220/tau2"
    config.log_dir = f"logs/{info}"
    config.algo = "comfo5"
    config.model_dir = f"saved_models/{info}"
    config.initializer = "orthogonal"
    config.hidden_dims = (256, 256)
    config.lr = 3e-4
    config.seed = 0
    config.tau = 0.005
    config.gamma = 0.99
    config.alpha = 5.0
    config.expectile = 0.7
    config.temperature = 3.0
    config.batch_size = 256
    config.eval_episodes = 10
    config.num_random = 10
    config.eval_freq = 5000
    config.mle_alpha = 1.0
    config.max_timesteps = 1000000
    return config
