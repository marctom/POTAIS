#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import pickle
import random
from pathlib import Path

import numpy as np
import gym


class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale


def train(args, seed, num):
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    from baselines.common import tf_util as U
    from baselines import logger
    from baselines.ppo1 import mlp_policy
    from baselines.common import set_global_seeds
    from algorithm import learn

    logger.configure()
    workerseed = seed + 10000 * num
    set_global_seeds(workerseed)

    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space,
                                    ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)

    env = RewScale(gym.make(args.env), args.reward_scale)
    results = learn(env=env,
                    policy_fn=policy_fn,
                    max_timesteps=args.num_timesteps,
                    timesteps_per_actorbatch=2048,
                    clip_param=args.clip,
                    entcoeff=0.0,
                    optim_epochs=10,
                    optim_stepsize=args.lr,
                    optim_batchsize=64,
                    gamma=0.99,
                    lam=0.95,
                    alphas=args.alphas,
                    schedule=args.schedule,
                    return_mv_avg=args.return_mv_avg)
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, type=str)
    parser.add_argument('--num_seeds', default=1, type=int)
    parser.add_argument('--num_timesteps', required=True, type=int)
    parser.add_argument('--alphas', required=True, type=str)
    parser.add_argument('--reward_scale', default=1, type=float)
    parser.add_argument('--clip', default=0.2, type=float)
    parser.add_argument('--return_mv_avg', default=200, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--schedule', default='linear', type=str,
                        help='Schedule for learning rate, linear or constant')

    args = parser.parse_args()
    args.alphas = [float(b) for b in args.alphas.split(' ')]

    tasks = [(args, random.randint(0, 2 ** 32), i)
             for i in range(args.num_seeds)]

    with mp.Pool(args.num_seeds) as pool:
        results = pool.starmap(train, tasks)

    Path('results').mkdir(exist_ok=True)
    name = 'results/results_{}_alphas-{}.pkl'.format(
        args.env, '--'.join(map(str, args.alphas)))
    with open(name, 'wb') as f:
        minlen = min(len(r) for r in results)
        results = np.vstack([r[:minlen] for r in results]) / args.reward_scale
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
