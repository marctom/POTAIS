import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--alphas', required=True, type=str)

args = parser.parse_args()

timesteps_dict = {
    "Swimmer-v3": 3e6,
    "Ant-v3": 3e6,
    "Walker2d-v3": 4e6,
    "Humanoid-v3": 10e6,
}

for env in sorted(timesteps_dict.keys()):
    if env == "Humanoid-v3":
        subprocess.call(["python", "run.py",
                         "--env", env,
                         "--num_seeds", "8",
                         "--num_timesteps", str(int(timesteps_dict[env])),
                         "--alphas", args.alphas,
                         "--clip", "0.2",
                         ])
    else:
        subprocess.call(["python", "run.py",
                         "--env", env,
                         "--num_seeds", "8",
                         "--num_timesteps", str(int(timesteps_dict[env])),
                         "--alphas", args.alphas,
                         "--clip", "0.4",
                         ])
