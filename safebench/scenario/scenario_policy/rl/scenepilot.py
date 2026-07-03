# safebench/scenario/policy/scenepilot_policy.py
# Date: 2025-10-18
# Aligns ScenePilot with Safebench scenario policy interface (like SAC/PPO)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

import math

from safebench.util.torch_util import CUDA, CPU
from safebench.scenario.scenario_policy.base_policy import BasePolicy


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(hid, action_dim)
        self.fc_std = nn.Linear(hid, action_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.min_std = 1e-3

    def forward(self, x):
        h = self.net(x)
        # mu  = self.tanh(self.fc_mu(h))   
        mu  = self.tanh(self.fc_mu(h))                         # [-1,1]
        std = self.softplus(self.fc_std(h)) + self.min_std       # (0,+)
        return mu, std

    def sample_action(self, x, deterministic=False):
        mu, std = self(x)
        if deterministic:
            a = mu
        else:
            a = Normal(mu, std).sample()
        return a.clamp_(-1.0, 1.0)
    
class ValueNet(nn.Module):
    def __init__(self, state_dim, hid=128, head_hid=64):
        """
        Shared encoder with separate head MLPs:
          shared_trunk: f_theta(s) -> h
          head_risk   : g_r(h) -> V_risk(s)
          head_sigma  : g_s(h) -> V_sigma(s)

        forward returns [B, 2], where [:, 0] is V_risk and [:, 1] is V_sigma.
        """
        super().__init__()
        # Shared scenario encoder.
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        # risk  head
        self.head_risk = nn.Sequential(
            nn.Linear(hid, head_hid), nn.ReLU(),
            nn.Linear(head_hid, 1),
        )
        # sigma head
        self.head_sigma = nn.Sequential(
            nn.Linear(hid, head_hid), nn.ReLU(),
            nn.Linear(head_hid, 1),
        )

    def forward(self, x):
        """
        x: [B, state_dim]

        V: [B, 2]  (V[:,0]=V_risk, V[:,1]=V_sigma)
        """
        h = self.shared(x)                     # [B, hid]

        v_r = self.head_risk(h)    # [B, 1]
        v_s = self.head_sigma(h)   # [B, 1]

        V = torch.cat([v_r, v_s], dim=-1)      # [B, 2]
        return V



class ScenePilotBase(BasePolicy):
    """
    Scenario-policy ScenePilot (on-policy), aligned with the SAC/PPO interface:
      - info_process() reads actor_info from infos only
      - get_init_action() returns [None] * B
      - get_action() returns a tensor/ndarray, not a dict, for direct env use
      - train() reads actor_info/n_actor_info and 2D reward [risk, sigma] from the replay buffer
    """
    name = 'ScenePilot'
    type = 'onpolicy'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.logger = logger
        algo_cfg = config.get('scenepilot')
        if algo_cfg is None:
            raise KeyError("Scenario config must define 'scenepilot'.")
        ipo = algo_cfg['ipo']

        # Read dimensions from scenario config, consistent with SAC.
        self.state_dim  = int(config['scenario_state_dim'])
        self.action_dim = int(config['scenario_action_dim'])

        # ------ Training hyperparameters ------
        self.lr         = float(algo_cfg.get('lr', 3e-4))
        self.batch_size = int(ipo.get('batch_size', 2048))
        self.epochs     = int(ipo.get('ppo_epochs', 4))
        self.ent_coef   = float(ipo.get('entropy_coef', 0.0))
        self.clip_eps   = float(algo_cfg.get('ppo_clip', 0.2))
        self.feas_tol   = float(algo_cfg.get('feas_tol', 0.0))

        # Threshold cycles through k_values only.
        raw_k_values = algo_cfg.get('k_values')
        if not isinstance(raw_k_values, (list, tuple)) or len(raw_k_values) == 0:
            raise ValueError("scenepilot.k_values must be a non-empty list of numbers.")
        self.k_values = [float(x) for x in raw_k_values]
        self.k_index = 0
        self.k = self.k_values[self.k_index]
        self._k_pending = None  # New k value to apply on the next episode.
        self._k_pending_index = None

        # ------ Model and actor optimizer ------
        self.policy = CUDA(PolicyNet(self.state_dim, self.action_dim))
        self.optim  = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # ------ Value network estimates long-term returns for risk and sigma. ------
        self.gamma     = float(algo_cfg.get('gamma', 0.99))
        self.lam       = float(algo_cfg.get('gae_lambda', 0.95)) 
        self.value     = CUDA(ValueNet(self.state_dim, hid=128, head_hid=64))
        self.v_optim   = torch.optim.Adam(self.value.parameters(), lr=self.lr)

        # Paths and resume state, following the scenario-policy style.
        self.model_id    = config['model_id']
        self.scenario_id = config['scenario_id']
        self.route_id    = config['route_id']
        self.model_path  = os.path.join(config['ROOT_DIR'], config['model_path'])
        os.makedirs(self.model_path, exist_ok=True)
        self.continue_episode = 0

        self.mode = 'train'
        self.gated = bool(algo_cfg.get('gated', False))

        self.thresh_update_every = int(algo_cfg.get('thresh_update_every', 50))
        self._thresh_tick = 0

    # ---------- Mode switch ----------
    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy.train()
        elif mode == 'eval':
            self.policy.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

        
    def _gae_1d(self, deltas: torch.Tensor, dones: torch.Tensor):
        """
        deltas: [T] = r_t + γ V(s_{t+1}) - V(s_t)
        dones : [T]  1 means this step ends the episode.
        Multiple concatenated episodes are allowed; dones breaks the GAE recursion.
        """
        T = deltas.shape[0]
        adv = torch.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]      # Clear GAE recursion where done=1.
            gae = deltas[t] + self.gamma * self.lam * mask * gae
            adv[t] = gae
        return adv


    # ---------- State extraction shared with SAC/PPO ----------
    def info_process(self, infos):
        """
        infos: List[dict], each item must contain 'actor_info'
        Returns [B, scenario_state_dim].
        """
        x = np.stack([i['actor_info'] for i in infos], axis=0)
        x = x.reshape(x.shape[0], -1)
        # Align dimensions defensively.
        if x.shape[1] != self.state_dim:
            if x.shape[1] > self.state_dim:
                x = x[:, :self.state_dim]
            else:
                pad = np.zeros((x.shape[0], self.state_dim - x.shape[1]), dtype=x.dtype)
                x = np.concatenate([x, pad], axis=1)
        return x

    # ---------- Runner interface: initial action ----------
    def get_init_action(self, state, deterministic=False):
        num_scenario = len(state)
        additional_in = {}
        return [None] * num_scenario, additional_in

    # ---------- Runner interface: step action ----------
    def get_action(self, state, infos, deterministic=False):
        """
        Matches SAC/PPO by ignoring state and deriving actor_info from infos.
        Returns a CPU tensor/ndarray for VectorWrapper to pass directly to env.
        """
        S = CUDA(torch.as_tensor(self.info_process(infos), dtype=torch.float32))
        a = self.policy.sample_action(S, deterministic)
        return CPU(a)

    # ---------- Training (IPO on-policy) ----------
    def train(self, replay_buffer, ep_stats=None):

        # Apply pending k at the start of training so checkpoints stay under the active k directory.
        if self._k_pending is not None:
            old_k = float(self.k)
            new_k = float(self._k_pending)
            new_k_index = self._k_pending_index
            self._k_pending = None  # Clear pending value.
            self._k_pending_index = None

            if new_k_index is not None:
                self.k_index = new_k_index

            # Use numeric comparison to detect a real change.
            if not math.isclose(new_k, old_k, rel_tol=0.0, abs_tol=1e-12):
                if self.logger:
                    self.logger.log(f"[{self.name}] apply deferred k switch: {old_k:.4f} -> {new_k:.4f}", "cyan")
                self.k = new_k

                # Check whether the new k directory already has a checkpoint.
                k_dir_path = os.path.join(self.model_path, str(self.scenario_id), str(self.route_id), f"{self.k:.4f}")
                has_ckpt = False
                if os.path.isdir(k_dir_path):
                    try:
                        for n in os.listdir(k_dir_path):
                            if self._is_ckpt_filename(n):
                                has_ckpt = True
                                break
                    except Exception:
                        pass

                if has_ckpt:
                    # If present, load the latest checkpoint for that k.
                    self.load_model(None)
                else:
                    # If absent, keep weights and reset optimizer momentum for faster adaptation.
                    self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # 1) Record episode-level statistics.
        if ep_stats:
            J_sigma = float(np.mean([x['J_sigma'] for x in ep_stats])) if len(ep_stats) else None
            J_risk  = float(np.mean([x['J_risk']  for x in ep_stats])) if len(ep_stats) else None
            if J_sigma is not None:
                if self.logger:
                    self.logger.add_training_results('J_sigma', J_sigma)
                    self.logger.add_training_results('J_risk',  J_risk)
                    self.logger.add_training_results('k', self.k)

        # Keep rollout order stable for the step-wise shielding gate.
        # batch = replay_buffer.sample(self.batch_size)

        batch = replay_buffer.sample_scenepilot_rollout()
        B = batch['reward'].shape[0]    # Rollout length T

        # RouteReplayBuffer.sample() already provides actor_info, n_actor_info, done, reward, sigma_t, and sigma_tp1.
        # S   = CUDA(torch.as_tensor(batch['actor_info'],   dtype=torch.float32)).reshape(self.batch_size, -1)
        # S_n = CUDA(torch.as_tensor(batch['n_actor_info'], dtype=torch.float32)).reshape(self.batch_size, -1)
        # A   = CUDA(torch.as_tensor(batch['action'],       dtype=torch.float32)).reshape(self.batch_size, -1)

        S   = CUDA(torch.as_tensor(batch['actor_info'],   dtype=torch.float32)).reshape(B, -1)
        S_n = CUDA(torch.as_tensor(batch['n_actor_info'], dtype=torch.float32)).reshape(B, -1)
        A   = CUDA(torch.as_tensor(batch['action'],       dtype=torch.float32)).reshape(B, -1)
        # reward: [risk, sigma]
        Rv  = CUDA(torch.as_tensor(batch['reward'],       dtype=torch.float32))   # [:,0]=risk, [:,1]=sigma
        r_r  = Rv[:, 0]   # Immediate reward for the risk objective
        r_si = Rv[:, 1]   # Immediate reward for the sigma objective (physical margin)
        done = CUDA(torch.as_tensor(batch['done'],        dtype=torch.float32)).reshape(-1)

        sig_t   = CUDA(torch.as_tensor(batch['sigma_t'],   dtype=torch.float32)).reshape(-1)
        sig_tp1 = CUDA(torch.as_tensor(batch['sigma_tp1'], dtype=torch.float32)).reshape(-1)
        

        # 4) Two-head value network learns V_r and V_sigma with TD(0).
        V      = self.value(S)        # [B, 2]
        V_next = self.value(S_n).detach()   # [B, 2], no grad

        V_r      = V[:, 0]
        V_sigma  = V[:, 1]
        V_r_next = V_next[:, 0]
        V_s_next = V_next[:, 1]

        # TD target: target = r + γ (1-done) V(s')
        target_r = (r_r  + self.gamma * (1.0 - done) * V_r_next).detach()
        target_s = (r_si + self.gamma * (1.0 - done) * V_s_next).detach()

        # ======  h_t / h_{t+1} step-level shielding  α_t, last step and current -- 
        thresh = self._sigma_threshold_tensor().to(sig_t.device)   # Scalar threshold
        h_t, h_tp1 = torch.clamp(thresh - sig_t,   min=0.0), torch.clamp(thresh - sig_tp1, min=0.0)
        

        # 6) Unsafe -> A_f; safe and next step safe -> A_r; safe but next step unsafe -> A_f.
        unsafe_t   = (h_t   > 0).float()
        unsafe_tp1 = (h_tp1 > 0).float()
        safe_t, safe_tp1 = 1.0 - unsafe_t, 1.0 - unsafe_tp1

        # alpha_t: 1 means feasibility dominates; 0 means risk dominates.
        # Logic: unsafe -> full feasibility; safe and next step safe -> risk;
        #       safe but next step becomes unsafe -> early feasibility.
        alpha = unsafe_t + safe_t * unsafe_tp1    # [B], ∈ {0,1}


        # only consider the current state
        # h_t = torch.clamp(thresh - r_si,   min=0.0)
        # unsafe_t   = (h_t   > 0).float()

        # Compute mixed value.
        # # 6) Mixed critic: V_mix(s) = (1-α) V_r + α V_sigma -- 
        # V_mix      = (1.0 - alpha) * V_r     + alpha * V_sigma
        # target_mix = (1.0 - alpha) * target_r + alpha * target_s

        # # TD residual: delta_mix = target_mix - V_mix
        # deltas_mix = target_mix - V_mix

        # # 7) Use GAE on V_mix to get an advantage.
        # with torch.no_grad():
        #     # This assumes the batch is approximately time-concatenated; for strict per-episode computation,
        #     # change this later to a trajectory-sampling version.
        #     A_mix_raw = self._gae_1d(deltas_mix, done)
        #     ADV = (A_mix_raw - A_mix_raw.mean()) / (A_mix_raw.std() + 1e-6)


        # 6) Mixed advantage
        # TD residual: delta_mix = target_mix - V_mix
        deltas_V_r = target_r - V_r
        deltas_V_sigma = target_s - V_sigma

        with torch.no_grad():
            # This assumes the batch is approximately time-concatenated; for strict per-episode computation,
            # change this later to a trajectory-sampling version.
            A_V_r = self._gae_1d(deltas_V_r, done)
            # ADV_r = (A_V_r - A_V_r.mean()) / (A_V_r.std() + 1e-6)
            ADV_r = (A_V_r - A_V_r.mean()) / (A_V_r.std() + 1e-6)

            A_V_sigma = self._gae_1d(deltas_V_sigma, done)
            # ADV_sigma = (A_V_sigma - A_V_sigma.mean()) / (A_V_sigma.std() + 1e-6)
            ADV_sigma = (A_V_sigma - A_V_sigma.mean()) / (A_V_sigma.std() + 1e-6)

            ADV = (1.0 - alpha) * ADV_r + alpha * ADV_sigma

            # Old-policy log probability for the PPO ratio.
            with torch.no_grad():
                old_mu, old_std = self.policy(S)
                old_dist = Normal(old_mu, old_std)
                old_logp = old_dist.log_prob(A).sum(-1)

        # ====== 8) Multi-epoch PPO updates for policy and value. ======
        batch_size = min(self.batch_size, B)
        total_policy_loss, total_value_loss, update_steps = 0.0, 0.0, 0

        for _ in range(self.epochs):
            idx = torch.randperm(B, device=S.device)
            for start in range(0, B, batch_size):
                mb_idx = idx[start:start + batch_size]

                mb_S = S[mb_idx]
                mb_A = A[mb_idx]
                mb_ADV = ADV[mb_idx]
                mb_old_logp = old_logp[mb_idx]

                # Value update
                mb_V = self.value(mb_S)
                mb_V_r = mb_V[:, 0]
                mb_V_sigma = mb_V[:, 1]
                mb_target_r = target_r[mb_idx].detach()
                mb_target_s = target_s[mb_idx].detach()

                value_loss = F.mse_loss(mb_V_r, mb_target_r) + F.mse_loss(mb_V_sigma, mb_target_s)
                self.v_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.v_optim.step()

                # Policy update (PPO clip)
                mu, std = self.policy(mb_S)
                dist = Normal(mu, std)
                logp = dist.log_prob(mb_A).sum(-1)
                entropy = dist.entropy().sum(-1)

                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_ADV
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_ADV
                policy_loss = -(torch.min(surr1, surr2).mean() + self.ent_coef * entropy.mean())

                self.optim.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optim.step()

                total_policy_loss += float(policy_loss.detach().cpu())
                total_value_loss += float(value_loss.detach().cpu())
                update_steps += 1

        # ====== 9) Threshold scheduling and statistics ======
        feas_ratio = (sig_tp1 >= float(thresh)).float().mean().item()
        if self.logger:
            self.logger.add_training_results('feasible_ratio', feas_ratio)
            self.logger.add_training_results('thresh',      float(thresh.item()))
            if update_steps > 0:
                self.logger.add_training_results('value_loss',  total_value_loss / update_steps)
                self.logger.add_training_results('policy_loss', total_policy_loss / update_steps)
        self._advance_threshold(feasible=(feas_ratio > 0.5))

        replay_buffer.reset_buffer()
        return

    # ---------- Threshold helpers ----------
    def _sigma_threshold_tensor(self):
        return torch.tensor(self.k, dtype=torch.float32)


    def _advance_threshold(self, feasible: bool):

        # Absolute threshold mode (epsilon = k): advance to the next k_values entry by window.
        self._thresh_tick += 1

        # Skip until the cooldown window is reached.
        if self._thresh_tick < self.thresh_update_every:
            return
        
        new_k_index = (self.k_index + 1) % len(self.k_values)
        new_k_val = self.k_values[new_k_index]
        
        self._thresh_tick = 0

        # Do not switch immediately; apply it on the next episode.
        self._k_pending = new_k_val
        self._k_pending_index = new_k_index


    def sigma_threshold_value(self) -> float:
        """
        Return the threshold currently in effect.
        """
        if self._k_pending is not None:
            return float (self._k_pending)
        else:
            return float(self._sigma_threshold_tensor().item())

    def _is_ckpt_filename(self, filename):
        return filename.startswith('model.scenepilot.') and filename.endswith('.torch')

    def _candidate_ckpt_names(self, episode):
        return [f'model.scenepilot.{self.model_id}.{episode:04}.torch']

    def _resolve_existing_path(self, base_dir, names):
        for name in names:
            path = os.path.join(base_dir, name)
            if os.path.isfile(path):
                return path
        return None

    # ---------- Persistence ----------
    def save_model(self, episode):
        states = {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optim': self.optim.state_dict(),
            'v_optim': self.v_optim.state_dict(),
        }
        k_dir = f"{self.k:.4f}"
        save_dir = os.path.join(self.model_path, str(self.scenario_id), str(self.route_id),k_dir)
        os.makedirs(save_dir, exist_ok=True)
        # always keep the max episode from runner
        f = os.path.join(save_dir, f'model.scenepilot.{self.model_id}.{episode:04}.torch')
        self.logger.log(f'>> Saving scenario policy {self.name} model to {f}')
        torch.save(states, f)

    def load_model(self, arg=None):
        """
        Supports two call patterns:
        1) load_model(scenario_configs: list)  - Evaluation loads by scenario file name
        2) load_model(episode: int|None)      - Training/resume loads the latest checkpoint by episode
        """
        # Load by scenario
        if isinstance(arg, list):
            scenario_configs = arg
            for cfg in scenario_configs:
                model_file = cfg.parameters
                model_dir = os.path.join(self.model_path, str(self.scenario_id))
                model_path = self._resolve_existing_path(model_dir, [model_file])
                if model_path is not None and os.path.isfile(model_path):
                    self.logger.log(f'>> Loading {self.name} model from {model_path}')
                    ckpt = torch.load(model_path, map_location='cpu')
                    self.policy.load_state_dict(ckpt['policy'])
                else:
                    fallback = os.path.join(model_dir, model_file)
                    self.logger.log(f'>> Fail to find {self.name} model at {fallback}', color="yellow")
            return

        # Load by episode; None finds the latest checkpoint automatically.
        episode = arg
        k_dir = f"{self.k:.4f}"
        if episode is None:
              # Filter checkpoints for the current k and model_id within the current route directory.
            save_dir_cur = os.path.join(self.model_path, str(self.scenario_id), str(self.route_id),k_dir)
            # prefix = f'model.scenepilot.{str(self.k)}.{self.model_id}.'
            episode = -1
            # for _, _, files in os.walk(self.model_path):
            #  load the max ep number in path, but don't save it as this ep No. + 1
            for _, _, files in os.walk(save_dir_cur):
                for n in files:
                    if self._is_ckpt_filename(n):
                        try:
                            ep = int(n.split('.')[-2])
                            episode = max(episode, ep)
                        except:
                            pass
        save_dir = os.path.join(self.model_path, str(self.scenario_id), str(self.route_id), k_dir)
        f = self._resolve_existing_path(save_dir, self._candidate_ckpt_names(episode))
        if f is None:
            f = os.path.join(save_dir, f'model.scenepilot.{self.model_id}.{episode:04}.torch')
        if os.path.isfile(f):
            self.logger.log(f'>> Loading scenario policy {self.name} model from {f}')
            ckpt = torch.load(f, map_location='cpu')
            self.policy.load_state_dict(ckpt['policy'])
            self.value.load_state_dict(ckpt['value'])
            self.optim.load_state_dict(ckpt['optim'])
            self.v_optim.load_state_dict(ckpt['v_optim'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No scenario policy {self.name} model found at {f}', 'yellow')


class ScenePilot(ScenePilotBase):
    name = 'ScenePilot'
