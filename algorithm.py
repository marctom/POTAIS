"""
Implementation adopted from https://github.com/openai/baselines
"""
from collections import deque
import time

import tensorflow as tf
import numpy as np

from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
from baselines.ppo1.pposgd_simple import traj_segment_generator


class AdamOptimizer:
    def __init__(self, var_list, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)

    def update(self, g, stepsize):
        g = g.astype('float32')
        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2 ** self.t) / (
                1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)


def learn(env, policy_fn, *,
          timesteps_per_actorbatch,
          clip_param,
          entcoeff,
          optim_epochs,
          optim_stepsize,
          optim_batchsize,
          gamma,
          lam,
          max_timesteps,
          alphas,
          schedule,
          return_mv_avg,
          adam_epsilon=1e-5):
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])
    new = tf.placeholder(dtype=tf.float32, shape=[None])
    ret = tf.placeholder(dtype=tf.float32, shape=[None])

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    log_ratio = pi.pd.logp(ac) - oldpi.pd.logp(ac)

    bool_new = tf.cast(new, tf.bool)

    def get_shift(i):
        shift = tf.concat([tf.zeros((i,)), log_ratio[:-i]], 0)
        shift = tf.where(bool_new, tf.zeros_like(shift), shift)
        for _ in range(1, i):
            shift = tf.where(tf.concat([tf.ones((i,), dtype=tf.bool), bool_new[:-i]], 0),
                             tf.zeros_like(shift), shift)
        return shift

    shifts = [log_ratio] + [get_shift(i) for i in range(1, len(alphas))]
    is_log_ratio = sum(b * s for b, s in zip(alphas, shifts))

    ratio = tf.exp(is_log_ratio)
    surr1 = ratio * atarg
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult, new],
                             losses + [U.flatgrad(total_loss, var_list)])
    adam = AdamOptimizer(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(
            oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, new], losses)

    U.initialize()

    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch,
                                     stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    lenbuffer = deque(maxlen=200)
    rewbuffer = deque(maxlen=return_mv_avg)
    results = []

    while True:
        if timesteps_so_far > max_timesteps:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError()

        logger.log("********** Iteration %i ************" % iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg[
            "tdlamret"]
        vpredbefore = seg["vpred"]
        atarg = (atarg - atarg.mean()) / atarg.std()
        d = Dataset(
            dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, new=seg["new"]))
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)

        assign_old_eq_new()
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        for _ in range(optim_epochs):
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"],
                                            batch["atarg"], batch["vtarg"],
                                            cur_lrmult, batch["new"])
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"],
                                       batch["vtarg"], cur_lrmult, batch["new"])
            losses.append(newlosses)

        meanlosses = np.vstack(losses).mean(0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_" + name, lossval)
        logger.record_tabular("ev_tdlam_before",
                              explained_variance(vpredbefore, tdlamret))
        listoflrpairs = [(seg["ep_lens"], seg["ep_rets"])]
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        results.append(np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("AvgEpisodeLen", np.mean(lenbuffer))
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed [h]", (time.time() - tstart) / 3600)
        logger.record_tabular("Timesteps/sec",
                              timesteps_so_far / (time.time() - tstart))
        logger.dump_tabular()
    return results


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
