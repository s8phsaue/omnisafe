# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the CSC algorithm."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.common.lagrange import Lagrange
from omnisafe.models.actor_critic.csc_actor_critic import CSCActorCritic
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)

@registry.register
class CSC(TRPO):
    """The Conservative Safety Critics (CSC) algorithm.

    CSC is a safe RL derivative of TRPO.

    References:
        - Title: Conservative Safety Critics for Exploration
        - Authors: Homanga Bharadhwaj, Aviral Kumar, Nicholas Rhinehart, Sergey Levine, Florian Shkurti, Animesh Garg.
        - URL: `CSC <https://arxiv.org/abs/2010.14497>`_
        - Code Author: Philipp Sauer
    """

    def _init(self) -> None:
        """The initialization of the algorithm.

        Here we additionally initialize the Lagrange multiplier.
        """
        super()._init()
        self._check_algo_cfgs()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        """Log the Conservative Safety Critics specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        | Value/Adv_Cost             | The cost advantage.      |
        +----------------------------+--------------------------+
        | Misc/T_Cost                | The safety threshold.    |
        +----------------------------+--------------------------+
        

        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('Value/Adv_Cost')
        self._logger.register_key('Misc/T_Cost')
    
    def _init_model(self) -> None:
        """Initialize the CSC actor critic model."""
        self._actor_critic: CSCActorCritic = CSCActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )
    
    def _check_algo_cfgs(self) -> None:
        assert self._cfgs.algo_cfgs.use_cost == True, "cost critic is required."
        assert 0 < self._cfgs.algo_cfgs.step_frac_init < 1, "step_frac_init must be between 0 and 1."
        
    def _csc_search_step(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        step_direction: torch.Tensor,
        theta_old: torch.Tensor,
        p_dist: torch.distributions.Distribution,
        total_steps: int = 20,
        step_frac: float = 0.7,
    ) -> torch.Tensor:
        r"""Use line-search to find the step size that satisfies the kl constraint.

        Args:
            obs (torch.Tensor): The observation.
            step_direction (torch.Tensor): The step direction.
            theta_old (torch.Tensor): The old policy parameters.
            p_dist (torch.distributions.Distribution): The old policy distribution.
            total_steps (int, optional): The total steps to search. Defaults to 20.
            step_frac (float, optional): The initial step fraction. Defaults to 0.7.

        Returns:
            A tuple of final step direction and the size of acceptance steps.
        """
        final_kl = 0.0
        for step in range(total_steps):
            theta_new = theta_old + step_frac * step_direction
            set_param_values_to_model(self._actor_critic.actor, theta_new)
            acceptance_step = step + 1

            with torch.no_grad():
                q_dist = self._actor_critic.actor(obs)
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean() # or .sum(-1, keepdim=True).mean()?
                kl = distributed.dist_avg(kl).mean()

            if not torch.isfinite(kl):
                self._logger.log('WARNING: KL not finite')
                continue
            elif kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'INFO: violated KL constraint {kl.item()} at step {acceptance_step}.')
            else:
                # step only if we are within the trust region
                self._logger.log(f'Accept step at i={acceptance_step}')
                final_kl = kl.item()
                break

            step_frac *= (1 - step_frac)**(acceptance_step)
        
        else:
            # if we didn't find a step that satisfies the kl constraint
            self._logger.log('INFO: no suitable step found...')
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0
        
        set_param_values_to_model(self._actor_critic.actor, theta_old)

        self._logger.store(
            {
                'Train/KL': final_kl,
            },
        )

        return step_frac * step_direction, acceptance_step

    def _compute_adv_surrogate(
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute surrogate loss.

        CSC uses the following surrogate loss:

        .. math::
        
            L = A^{R}_{\pi_{\theta}} (s, a) - \lambda * A^C_{\pi_{\theta}} (s, a) / (1 - \gamma)

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function of reward to update policy network.
        """
        ratio = self._lagrange.lagrangian_multiplier.item() / (1 - self._cfgs.algo_cfgs.cost_gamma)
        return (adv_r - ratio * adv_c)
    
    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network using the conjugate gradient algorithm following the steps:

        - Compute the gradient of the policy.
        - Compute the step direction.
        - Search for a step size that satisfies the kl constraint.
        - Update the policy network.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The reward advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.
        """
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv)
        p_dist = self._actor_critic.actor(obs)

        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = torch.dot(x, self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        beta = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
        step_direction = beta * x
        assert torch.isfinite(step_direction).all(), 'step_direction is not finite'

        step_direction, accept_step = self._csc_search_step(
            obs=obs, 
            step_direction=step_direction,
            theta_old=theta_old, 
            p_dist=p_dist,
            total_steps=self._cfgs.algo_cfgs.line_search_steps,
            step_frac=self._cfgs.algo_cfgs.step_frac_init,
        )

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss = self._loss_pi(obs, act, logp, adv)

        self._logger.store(
            {
                'Loss/Loss_pi': loss.item(),
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': beta.item(),  # called beta in the csc paper
                'Misc/FinalStepNorm': torch.norm(step_direction).mean().item(),
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/H_inv_g': x.norm().item(),
            }
        )

    def _update_reward_critic(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:
        return super()._update_reward_critic(obs, target_value_r)

    def _update_cost_critic_cql(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        target_value_c: torch.Tensor
    ) -> None:
        """ Implements the cost critic update with CQL from 
        <https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py>

        Args:
            obs (torch.Tensor): The observation tensor.
            actions (torch.Tensor): The action tensor.
            next_obs (torch.Tensor): The next observation tensor.
            target_value_c (torch.Tensor): The target value tensor.
        """
        
        batch_size = obs.shape[0]
        action_size = self._env.action_space.shape[-1]
        num_actions = self._cfgs.algo_cfgs.cql_n_actions
        
        temp_obs = obs.unsqueeze(1).repeat(1, num_actions, 1)
        temp_next_obs = next_obs.unsqueeze(1).repeat(1, num_actions, 1)
        
        cql_temp = self._cfgs.algo_cfgs.cql_temp
        cql_min_q_weight = self._cfgs.algo_cfgs.cql_min_q_weight

        # Regularized MSE Loss
        q = self._actor_critic.cost_critic(obs, actions)[0]
        q_loss = torch.nn.functional.mse_loss(q, target_value_c)   # TODO entropy regularization?

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                q_loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        # Add CQL
        random_actions = torch.empty(
            batch_size, num_actions, action_size, dtype=torch.float32, device=self._device
        ).uniform_(-1, 1)
        curr_actions, _, _, logp_curr = self._actor_critic.forward(temp_obs)
        next_actions, _, _, logp_next = self._actor_critic.forward(temp_next_obs)

        q_rand = self._actor_critic.cost_critic(temp_obs, random_actions)[0]
        q_curr = self._actor_critic.cost_critic(temp_obs, curr_actions)[0]
        q_next = self._actor_critic.cost_critic(temp_obs, next_actions)[0]

        random_density = np.log(0.5 ** action_size)
        cat_q = torch.cat(
            [q_rand - random_density, q_next - logp_next.detach(), q_curr - logp_curr.detach()], dim=1
        )
        min_q_loss = torch.logsumexp(cat_q / cql_temp, dim=1).mean()
        min_q_loss = min_q_loss * cql_min_q_weight * cql_temp
        min_q_loss = min_q_loss - q.mean() * cql_min_q_weight

        loss = q_loss + min_q_loss

        # Update network
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)

        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})
        
    
    # TODO maybe use a custom Lagrange class with custom loss instead
    def _update_lagrange_multiplier(self, Jc: float, adv_c: torch.Tensor) -> None:
        """ Update lagrange multiplier with a custom update that considers cost advantages.
        
        Args:
            Jc (float): The mean episode cost.
            adv_c (torch.Tensor): The cost advantage.

        """
        gamma = self._cfgs.algo_cfgs.cost_gamma
        cl = self._lagrange.cost_limit

        with torch.no_grad():
            grad = 1 / (1 - gamma) * adv_c.mean().cpu() - (cl - Jc)
            self._lagrange.lagrangian_multiplier.data -= self._lagrange.lambda_lr * grad
            self._lagrange.lagrangian_multiplier.data.clamp_(
                0.0,
                self._lagrange.lagrangian_upper_bound,
            )
        
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})
    
    def _get_update_data(self):
        # Retrieve default on-policy data
        data:dict = self._buf.get()
        data_additional = {
            'final_obs': [],
            'final_idx': [],
        }

        obs = data['obs']
        count = 0
        
        # Retrieve additional data
        for buf in self._buf.buffers:
            data_final = buf.get_final_data()
            final_obs = data_final['final_obs']
            final_idx = data_final['final_idx']

            inc = final_idx[-1].item()              # number of obs in the single buffer
            final_idx = (final_idx - 1) + count     # -1: shift to the left for next_obs, +count: as obs contains the concatenation of all buffers
            count += inc

            data_additional['final_obs'].append(final_obs)
            data_additional['final_idx'].append(final_idx)

        # Concatenate tensors
        data_additional = {k: torch.cat(v, dim=0) for k, v in data_additional.items()}
        final_obs, final_idx = (
            data_additional['final_obs'], 
            data_additional['final_idx'],
        )

        # Reconstruct next_obs
        next_obs = obs.detach().clone()
        next_obs[:-1] = obs[1:]
        next_obs[final_idx] = final_obs
        data['next_obs'] = next_obs
        del data_additional['final_obs'], data_additional['final_idx']

        # Return data
        data.update(data_additional)
        return data


    def _update(self) -> None:
        """Update actor, critic, cost critic and lagrange multiplier."""

        # Retrieve relevant data
        data = self._get_update_data()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c, next_obs = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
            data['next_obs'],
        )

        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'

        # Update actor first
        Tcost = (1 - self._cfgs.algo_cfgs.cost_gamma) * (self._lagrange.cost_limit - Jc)
        self._actor_critic.update_values(
            Tcost=Tcost,
        )
        self._update_actor(obs, act, logp, adv_r, adv_c)

        # Update critics second
        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, next_obs, target_value_r, target_value_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            for (
                obs,
                act,
                next_obs,
                target_value_r,
                target_value_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                self._update_cost_critic_cql(obs, act, next_obs, target_value_c)

        # Update lagrange multiplier last
        self._update_lagrange_multiplier(Jc, adv_c)

        # Store metrics
        self._logger.store(
            {
                'Train/StopIter': self._cfgs.algo_cfgs.update_iters,
                'Value/Adv': adv_r.mean().item(),
                'Value/Adv_Cost': adv_c.mean().item(),
                'Misc/T_Cost': Tcost,
            },
        )