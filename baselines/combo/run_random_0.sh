CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/mujoco.py --config.env_name=hopper-random-v2 --config.seed=0;
CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/mujoco.py --config.env_name=halfcheetah-random-v2 --config.seed=0;
CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/mujoco.py --config.env_name=halfcheetah-random-v2 --config.seed=1;
CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/mujoco.py --config.env_name=halfcheetah-random-v2 --config.seed=2;
CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/mujoco.py --config.env_name=halfcheetah-random-v2 --config.seed=3;
CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/mujoco.py --config.env_name=walker2d-random-v2 --config.seed=2;
CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/mujoco.py --config.env_name=walker2d-random-v2 --config.seed=3;
CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/mujoco.py --config.env_name=walker2d-random-v2 --config.seed=4;

