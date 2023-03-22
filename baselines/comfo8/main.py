from absl import app, flags
from ml_collections import config_flags
import os
import train


config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
FLAGS = flags.FLAGS


mle_alphas = {
    "halfcheetah-random-v2": 1.0,
    "halfcheetah-medium-v2": 0,
    "halfcheetah-medium-replay-v2": 0,
    "halfcheetah-medium-expert-v2": 1.0,
    "halfcheetah-expert-v2": 1.0,
    "hopper-random-v2": 1.0,
    "hopper-medium-v2": 1.0,
    "hopper-medium-replay-v2": 0,
    "hopper-medium-expert-v2": 1.0,
    "hopper-expert-v2": 1.0,
    "walker2d-random-v2": 1.0,
    "walker2d-medium-v2": 1.0,
    "walker2d-medium-replay-v2": 0,
    "walker2d-medium-expert-v2": 1.0,
    "walker2d-expert-v2": 1.0,
}
conservative_weight = {
    "halfcheetah-random-v2": 1.0,
    "halfcheetah-medium-v2": 1.0,
    "halfcheetah-medium-replay-v2": 1.0,
    "halfcheetah-medium-expert-v2": 1.0,
    "halfcheetah-expert-v2": 1.0,
    "hopper-random-v2": 1.0,
    "hopper-medium-v2": 0.1,
    "hopper-medium-replay-v2": 0.1,
    "hopper-medium-expert-v2": 1.0,
    "hopper-expert-v2": 1.0,
    "walker2d-random-v2": 1.0,
    "walker2d-medium-v2": 1.0,
    "walker2d-medium-replay-v2": 0.1,
    "walker2d-medium-expert-v2": 1.0,
    "walker2d-expert-v2": 1.0,
}
# noise_scale = {
#     "halfcheetah-medium-v2": 0.001,
#     "halfcheetah-medium-replay-v2": 0.001,
#     "halfcheetah-medium-expert-v2": 0.001,
#     "hopper-medium-v2": 0.0005,
#     "hopper-medium-replay-v2": 0.0005,
#     "hopper-medium-expert-v2": 0.0005,
#     "walker2d-medium-v2": 0.001,
#     "walker2d-medium-replay-v2": 0.001,
#     "walker2d-medium-expert-v2": 0.001,
# }


def main(argv):
    configs = FLAGS.config
    configs.mle_alpha = mle_alphas[configs.env_name]
    configs.conservative_weight = conservative_weight[configs.env_name]
    # configs.noise_scale = noise_scale[configs.env_name]
    os.makedirs(f"{configs.log_dir}/{configs.env_name}", exist_ok=True)
    os.makedirs(f"{configs.model_dir}/{configs.env_name}", exist_ok=True)
    train.train_and_evaluate(configs)


if __name__ == '__main__':
    app.run(main)
