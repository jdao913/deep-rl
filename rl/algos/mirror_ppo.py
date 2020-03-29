import time
from copy import deepcopy
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.utils.tensorboard import SummaryWriter
from ..utils.logging import Logger

import numpy as np
from rl.algos import PPO
from rl.envs import Vectorize, Normalize
import sys


# TODO:
# env.mirror() vs env.matrix?

# TODO: use magic to make this reuse more code (callbacks etc?)

class MirrorPPO(PPO):
    def update(self, policy, old_policy, optimizer,
               observations, actions, returns, advantages,
               env_fn
    ):
        env = env_fn()
        mirror_observation = env.mirror_observation
        if env.clock_based:
            mirror_observation = env.mirror_clock_observation
        mirror_action = env.mirror_action

        # Use only half of minibatch_size since mirror states will double the minibatch size
        minibatch_size = int(self.minibatch_size / 2) or advantages.numel()  
        print("minibatch_size / 2: ", minibatch_size)

        for _ in range(self.epochs):
                losses = []
                sampler = BatchSampler(
                    SubsetRandomSampler(range(advantages.numel())),
                    minibatch_size,
                    drop_last=True
                )
                for indices in sampler:
                    indices = torch.LongTensor(indices)

                    orig_obs = observations[indices]
                    # obs_batch = torch.cat(
                    #     [obs_batch,
                    #      obs_batch @ torch.Tensor(env.obs_symmetry_matrix)]
                    # ).detach()

                    action_batch = actions[indices]
                    # action_batch = torch.cat(
                    #     [action_batch,
                    #      action_batch @ torch.Tensor(env.action_symmetry_matrix)]
                    # ).detach()

                    return_batch = returns[indices]
                    # return_batch = torch.cat(
                    #     [return_batch,
                    #      return_batch]
                    # ).detach()

                    advantage_batch = advantages[indices]
                    # advantage_batch = torch.cat(
                    #     [advantage_batch,
                    #      advantage_batch]
                    # ).detach()

                    # Add mirror states to minibatch (only obs and actions need to mirrored)
                    if env.clock_based:
                        mir_obs = mirror_observation(orig_obs, env.clock_inds)
                    else:
                        mir_obs = mirror_observation(orig_obs)
                    mir_actions = mirror_action(action_batch)
                    
                    # print("action batch: ", action_batch)
                    # obs_batch = torch.cat([orig_obs, mir_obs])
                    # action_batch = torch.cat([action_batch, mir_actions])
                    # return_batch = torch.cat([return_batch, return_batch])
                    # advantage_batch = torch.cat([advantage_batch, advantage_batch])
                    
                    # print("mir_action: ", mir_actions)
                    # print("action batch: ", action_batch)
                    
                    obs_batch = orig_obs
                    values, pdf = policy.evaluate(obs_batch)

                    # TODO, move this outside loop?
                    with torch.no_grad():
                        _, old_pdf = old_policy.evaluate(obs_batch)
                        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)
                    
                    log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)
                    
                    ratio = (log_probs - old_log_probs).exp()

                    cpi_loss = ratio * advantage_batch
                    clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch
                    actor_loss = -torch.min(cpi_loss, clip_loss).mean()

                    critic_loss = 0.5 * (return_batch - values).pow(2).mean()

                    # Mirror Symmetry Loss
                    _, deterministic_actions = policy(orig_obs)
                    _, mirror_actions = policy(mir_obs)
                    mirror_actions = mirror_action(mirror_actions)

                    mirror_loss = 4 * (deterministic_actions - mirror_actions).pow(2).mean()

                    entropy_penalty = -self.entropy_coeff * pdf.entropy().mean()

                    # TODO: add ability to optimize critic and actor seperately, with different learning rates

                    optimizer.zero_grad()
                    (actor_loss + critic_loss + mirror_loss + entropy_penalty).backward()

                    # Clip the gradient norm to prevent "unlucky" minibatches from 
                    # causing pathalogical updates
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
                    optimizer.step()

                    losses.append([actor_loss.item(),
                                   pdf.entropy().mean().item(),
                                   critic_loss.item(),
                                   ratio.mean().item(),
                                   mirror_loss.item()])

                # TODO: add verbosity arguments to suppress this
                print(' '.join(["%g"%x for x in np.mean(losses, axis=0)]))

                # Early stopping 
                if kl_divergence(pdf, old_pdf).mean() > 0.02:
                    print("Max kl reached, stopping optimization early.")
                    break
        return np.mean(losses, axis=0)

    def train(self,
              env_fn,
              policy, 
              n_itr,
              normalize=None,
              logger=None):

        if normalize != None:
            policy.train()
        else:
            policy.train(0)

        env = Vectorize([env_fn]) # this will be useful for parallelism later
        
        if normalize is not None:
            env = normalize(env)

            mean, std = env.ob_rms.mean, np.sqrt(env.ob_rms.var + 1E-8)
            policy.obs_mean = torch.Tensor(mean)
            policy.obs_std = torch.Tensor(std)
            policy.train(0)
        
        opt_time = np.zeros(n_itr)
        samp_time = np.zeros(n_itr)
        eval_time = np.zeros(n_itr)

        old_policy = deepcopy(policy)

        optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)

        start_time = time.time()

        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            sample_start = time.time()
            batch = self.sample_parallel(env_fn, policy, self.num_steps, self.max_traj_len)

            print("time elapsed: {:.2f} s".format(time.time() - start_time))
            print("sample time elapsed: {:.2f} s".format(time.time() - sample_start))

            observations, actions, returns, values = map(torch.Tensor, batch.get())

            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.numel())

            old_policy.load_state_dict(policy.state_dict())  # WAY faster than deepcopy

            optimizer_start = time.time()

            losses = self.update(policy, old_policy, optimizer, observations, actions, returns, advantages, env_fn) 
           
            print("optimizer time elapsed: {:.2f} s".format(time.time() - optimizer_start))        

            evaluate_start = time.time()
            test = self.sample_parallel(env_fn, policy, 800 // self.n_proc, self.max_traj_len, deterministic=True)
            eval_time[itr] = time.time() - evaluate_start
            print("evaluate time elapsed: {:.2f} s".format(eval_time[itr]))
            
            if logger is not None:    

                avg_eval_reward = np.mean(test.ep_returns)
                avg_batch_reward = np.mean(batch.ep_returns)
                avg_ep_len = np.mean(batch.ep_lens)

                _, pdf     = policy.evaluate(observations)
                _, old_pdf = old_policy.evaluate(observations)

                entropy = pdf.entropy().mean().item()
                kl = kl_divergence(pdf, old_pdf).mean().item()

                grads = np.concatenate([p.grad.data.numpy().flatten() for p in policy.parameters() if p.grad is not None])

                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (test)', avg_eval_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (batch)', avg_batch_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', avg_ep_len) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % kl) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % entropy) + "\n")
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.flush()

                if type(logger) is Logger:
                    logger.record("Return (test)", np.mean(test.ep_returns))
                    logger.record("Return (batch)", np.mean(batch.ep_returns))
                    logger.record("Mean Eplen",  np.mean(batch.ep_lens))
            
                    logger.record("Mean KL Div", kl)
                    logger.record("Mean Entropy", entropy)
                    logger.dump()
                elif type(logger) is SummaryWriter:
                    logger.add_scalar("Data/Return (test)", avg_eval_reward, itr)
                    logger.add_scalar("Data/Return (batch)", avg_batch_reward, itr)
                    logger.add_scalar("Data/Mean Eplen", avg_ep_len, itr)

                    logger.add_scalar("Gradients Info/Grad Norm", np.sqrt(np.mean(np.square(grads))), itr)
                    logger.add_scalar("Gradients Info/Max Grad", np.max(np.abs(grads)), itr)
                    logger.add_scalar("Gradients Info/Grad Var", np.var(grads), itr)

                    # logger.add_scalar("Action Info/Max action", max_act, itr)
                    # logger.add_scalar("Action Info/Max mirror action", max_acts[1], itr)

                    logger.add_scalar("Misc/Mean KL Div", kl, itr)
                    logger.add_scalar("Misc/Mean Entropy", entropy, itr)
                    logger.add_scalar("Misc/Critic Loss", losses[2], itr)
                    logger.add_scalar("Misc/Actor Loss", losses[0], itr)
                    logger.add_scalar("Misc/Mirror Loss", losses[4], itr)

                    logger.add_scalar("Misc/Sample Times", samp_time[itr], itr)
                    logger.add_scalar("Misc/Optimize Times", opt_time[itr], itr)
                    logger.add_scalar("Misc/Evaluation Times", eval_time[itr], itr)
                else:
                    print("No Logger")

            # TODO: add option for how often to save model
            if np.mean(test.ep_returns) > self.max_return:
                self.max_return = np.mean(test.ep_returns)
                self.save(policy, env)

            print("Total time: {:.2f} s".format(time.time() - start_time))
