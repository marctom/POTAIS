import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--alphas', required=True, type=str)

args = parser.parse_args()

timesteps_dict = {
    "Reacher-v2": 1e6,
    "Swimmer-v3": 3e6,
    "Hopper-v3": 2e6,
    "HalfCheetah-v3": 2e6,
    "Walker2d-v3": 3e6,
    "Ant-v3": 4e6,
    "Humanoid-v3": 10e6,
    "HumanoidStandup-v2": 4e6
}

for env in sorted(timesteps_dict.keys()):
    if env == "Humanoid-v3":
        subprocess.call(["python", "run.py",
                         "--env", env,
                         "--num_seeds", "10",
                         "--num_timesteps", str(int(timesteps_dict[env])),
                         "--alphas", args.alphas,
                         "--clip", "0.1",
                         "--schedule", "constant",
                         "--lr", "0.0001",
                         "--reward_scale", "0.1",
                         "--return_mv_avg", "400"
                         ])
    else:
        subprocess.call(["python", "run.py",
                         "--env", env,
                         "--num_seeds", "20",
                         "--num_timesteps", str(int(timesteps_dict[env])),
                         "--alphas", args.alphas,
                         ])
