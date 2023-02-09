CUDA_VISIBLE_DEVICES=1 python main.py --config=configs/mujoco.py --config.env_name=walker2d-medium-replay-v2 --config.seed=0;
CUDA_VISIBLE_DEVICES=1 python main.py --config=configs/mujoco.py --config.env_name=walker2d-medium-replay-v2 --config.seed=1;
CUDA_VISIBLE_DEVICES=1 python main.py --config=configs/mujoco.py --config.env_name=hopper-medium-replay-v2 --config.seed=1;
CUDA_VISIBLE_DEVICES=1 python main.py --config=configs/mujoco.py --config.env_name=hopper-medium-replay-v2 --config.seed=2;
CUDA_VISIBLE_DEVICES=1 python main.py --config=configs/mujoco.py --config.env_name=hopper-medium-replay-v2 --config.seed=3;
CUDA_VISIBLE_DEVICES=1 python main.py --config=configs/mujoco.py --config.env_name=hopper-medium-replay-v2 --config.seed=4;
CUDA_VISIBLE_DEVICES=1 python main.py --config=configs/mujoco.py --config.env_name=halfcheetah-medium-replay-v2 --config.seed=4;
