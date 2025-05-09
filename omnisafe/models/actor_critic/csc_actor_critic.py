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

"""Implementation of CSCActorCritic."""

import torch
from torch import optim

from omnisafe.models.actor_critic.actor_critic import ActorCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig

class CSCActorCritic(ActorCritic):
    """CSCActorCritic is a wrapper around ActorCritic that deploys an additional cost critic as a shield. It deploys a value network to estimate rewards and an additional Q network to estimate costs.
        
        Args:
            obs_space (OmnisafeSpace): The observation space.
            act_space (OmnisafeSpace): The action space.
            model_cfgs (ModelConfig): The model configurations.
            epochs (int): The number of epochs.
        
        Attributes:
            actor (Actor): The actor network.
            target_actor (Actor): The target actor network.
            reward_critic (Critic): The critic network.
            target_reward_critic (Critic): The target critic network.
            cost_critic (Critic): The critic network.
            target_cost_critic (Critic): The target critic network.
            actor_optimizer (Optimizer): The optimizer for the actor network.
            reward_critic_optimizer (Optimizer): The optimizer for the critic network.
            cost_critic_optimizer (Optimizer): The optimizer for the critic network.
            std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """ Initialize an instance of :class:`CSCActorCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)
        self.cost_critic: Critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=model_cfgs.critic.num_critics,
            use_obs_encoder=True,
        ).build_critic('q')
        self.add_module('cost_critic', self.cost_critic)

        if model_cfgs.critic.cost_lr is not None:
            self.cost_critic_optimizer: optim.Optimizer
            self.cost_critic_optimizer = optim.Adam(
                self.cost_critic.parameters(),
                lr=model_cfgs.critic.cost_lr,
            )

        self._Tcost = 0.0
        self._n_shield_actions = model_cfgs.n_shield_actions
    
    def update_values(self, Tcost: float) -> None:
        self._Tcost = Tcost

    @torch.no_grad
    def step(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Choose the action based on the observation. Apply a shield to enforce safe actions according to our learned q safety critic.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            action0: The (safe) shielded action.
            value_r0: The reward value of the observation.
            value_c0: The cost value of the observation.
            log_prob0: The log probability of the action.
        """
        action0, value_r0, value_c0, log_prob0 = self.forward(obs, deterministic=deterministic)
        unsafe0 = value_c0 > self._Tcost    # shape: (batch_size,)

        # Shield actions
        if unsafe0.any():
            
            n_unsafe0 = unsafe0.int().sum()
            n_samples = self._n_shield_actions

            # Adjust obs
            obs = obs[unsafe0]
            obs = obs.repeat_interleave(n_samples, dim=0)  # shape: (n_unsafe0 * n_samples, obs_dim)

            # Sample all actions at once
            action, value_r, value_c, log_prob = self.forward(obs, deterministic=False)

            action = action.view(n_unsafe0, n_samples, action.shape[-1])    # shape: (n_unsafe0, n_samples, act_dim)
            value_r = value_r.view(n_unsafe0, n_samples)                    # shape: (n_unsafe0, n_samples)
            value_c = value_c.view(n_unsafe0, n_samples)                    # shape: (n_unsafe0, n_samples)
            log_prob = log_prob.view(n_unsafe0, n_samples)                  # shape: (n_unsafe0, n_samples)

            # Override first action with the one we already sampled
            action[:, 0] = action0[unsafe0]      # shape: (n_unsafe0, n_samples, act_dim)
            value_r[:, 0] = value_r0[unsafe0]    # shape: (n_unsafe0, n_samples)
            value_c[:, 0] = value_c0[unsafe0]    # shape: (n_unsafe0, n_samples)
            log_prob[:, 0] = log_prob0[unsafe0]  # shape: (n_unsafe0, n_samples)

            # Check for safety, get idxs of first (potentially) safe action within each row
            unsafe = value_c > self._Tcost      # shape: (n_unsafe0, n_samples)
            safe = ~unsafe
            m = safe.int().max(dim=1)
            first_safe_idx = m.indices          # shape: (n_unsafe0,)
            is_safe_action = m.values.bool()    # shape: (n_unsafe0,)
            is_unsafe_action = ~is_safe_action  # shape: (n_unsafe0,)
            
            # If there is a safe action, we use it (it is the first action we would have encountered by using a loop)
            if is_safe_action.any():
                mask0 = torch.zeros_like(unsafe0)   # shape: (batch_size,)
                mask0[unsafe0] = is_safe_action
                mask = torch.zeros_like(unsafe)     # shape: (n_unsafe0, n_samples)
                mask[torch.arange(n_unsafe0), first_safe_idx] = is_safe_action

                action0[mask0] = action[mask]
                value_r0[mask0] = value_r[mask]
                value_c0[mask0] = value_c[mask]
                log_prob0[mask0] = log_prob[mask]

            # Else we take the 'safest' action (with the minimum cost)
            if is_unsafe_action.any():
                mask0 = torch.zeros_like(unsafe0)   # shape: (batch_size,)
                mask0[unsafe0] = is_unsafe_action

                m = value_c[is_unsafe_action, :].min(dim=1)
                mask = torch.zeros_like(unsafe)     # shape: (n_unsafe0, n_samples)
                mask[is_unsafe_action.nonzero().flatten(), m.indices] = True

                action0[mask0] = action[mask]
                value_r0[mask0] = value_r[mask]
                value_c0[mask0] = value_c[mask]
                log_prob0[mask0] = log_prob[mask]
        
        return action0, value_r0, value_c0, log_prob0
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, ...]:
        """Choose the action based on the observation.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            tuple:
            - action (torch.Tensor): The deterministic action if ``deterministic`` is True, 
            otherwise the action with Gaussian noise.
            - value_r (float): The reward value of the observation.
            - value_c (float): The cost value of the observation.
            - log_prob (torch.Tensor): The log probability of the action.
        """
        with torch.no_grad():
            action = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(action)
            value_r = self.reward_critic(obs)
            value_c = self.cost_critic(obs, action)

        return action, value_r[0], value_c[0], log_prob
        
