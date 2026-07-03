''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 22:35:17
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import copy

import numpy as np
import carla
import pygame
from tqdm import tqdm

from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.gym_carla.replay_buffer import RouteReplayBuffer, PerceptionReplayBuffer

from safebench.agent import AGENT_POLICY_LIST
from safebench.scenario import SCENARIO_POLICY_LIST

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_data_loader import ScenarioDataLoader
from safebench.scenario.tools.scenario_utils import scenario_parse

from safebench.util.logger import Logger, setup_logger_kwargs
from safebench.util.metric_util import get_route_scores, get_perception_scores
from safebench.gym_carla.envs import av_safe  

import glob
import os


class CarlaRunner:
    def __init__(self, agent_config, scenario_config):
        self.scenario_config = scenario_config
        self.agent_config = agent_config
        # Hint for KING auto-generation path; used by scenario parser helper.
        self.scenario_config['agent_policy_type'] = agent_config.get('policy_type', '')
        self.scenario_config['safebench_agent_ckpt_dir_aim_bev'] = agent_config.get('aim_bev_ckpt_dir', None)
        self.scenario_config['safebench_agent_ckpt_dir_transfuser'] = agent_config.get('transfuser_ckpt_dir', None)
        self.scenario_config['safebench_agent_model_path'] = agent_config.get('model_path', None)
        self.rl_config = scenario_config['rl_config']

        self.seed = scenario_config['seed']
        self.exp_name = scenario_config['exp_name']
        self.output_dir = scenario_config['output_dir']
        self.mode = scenario_config['mode']
        self.save_video = scenario_config['save_video']

        self.render = scenario_config['render']
        self.num_scenario = scenario_config['num_scenario']
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']
        self.scenario_category = scenario_config['scenario_category']
        self.train_env_steps = scenario_config.get('train_env_steps', None)
        self.global_env_step = int(scenario_config.get('train_env_step_offset', 0) or 0)
        self.train_avsafe_steps = scenario_config.get('train_avsafe_steps', None)
        self.avsafe_step_offset = int(scenario_config.get('train_avsafe_step_offset', 0) or 0)
        self.max_wait_collision_episodes = scenario_config.get('max_wait_collision_episodes', None)
        self._last_avsafe_logged_update = -1

        # continue training flag
        self.continue_agent_training = scenario_config['continue_agent_training']
        self.continue_scenario_training = scenario_config['continue_scenario_training']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        # self.client = carla.Client('10.113.164.75', scenario_config['port'])
        self.client.set_timeout(30.0)
        self.world = None
        self.env = None

        self.env_params = {
            'auto_ego': scenario_config['auto_ego'],
            'obs_type': agent_config['obs_type'],
            'scenario_category': self.scenario_category,
            'ROOT_DIR': scenario_config['ROOT_DIR'],
            'warm_up_steps': 9,                                        # number of ticks after spawning the vehicles
            'disable_lidar': agent_config.get('disable_lidar', True),  # show bird-eye view lidar or not
            'display_size': 256,                                       # screen size of one bird-eye view window
            'obs_range': 32,                                           # observation range (meter)
            'd_behind': 12,                                            # distance behind the ego vehicle (meter)
            'max_past_step': 1,                                        # the number of past steps to draw
            'discrete': False,                                         # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],                          # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],                        # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],                     # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],                     # continuous steering angle range
            'max_episode_step': scenario_config['max_episode_step'],   # maximum timesteps per episode
            'max_waypt': 12,                                           # maximum number of waypoints
            'lidar_bin': 0.125,                                        # bin size of lidar sensor (meter)
            'out_lane_thres': 4,                                       # threshold for out of lane (meter)
            'desired_speed': 8,                                        # desired speed (m/s)
            'image_sz': 1024,                                          # TODO: move to config of od scenario
            'rl_config': self.rl_config,
            'route_level_avsafe_training': bool(
                scenario_config.get('route_level_avsafe_training', False)
                or self.train_avsafe_steps is not None
                or self.max_wait_collision_episodes is not None
            ),
        }

        # pass config from scenario to agent
        agent_config['mode'] = scenario_config['mode']
        agent_config['ego_action_dim'] = scenario_config['ego_action_dim']
        agent_config['ego_state_dim'] = scenario_config['ego_state_dim']
        agent_config['ego_action_limit'] = scenario_config['ego_action_limit']

        # define logger
        logger_kwargs = setup_logger_kwargs(
            self.exp_name, 
            self.output_dir, 
            self.seed,
            agent=agent_config['policy_type'],
            scenario=scenario_config['policy_type'],
            scenario_category=self.scenario_category
        )
        self.logger = Logger(**logger_kwargs)
        self.logger.init_wandb(
            use_wandb=bool(scenario_config.get('use_wandb', False)),
            project=scenario_config.get('wandb_project'),
            entity=scenario_config.get('wandb_entity'),
            name=scenario_config.get('wandb_name'),
            group=scenario_config.get('wandb_group'),
            mode=scenario_config.get('wandb_mode', 'online'),
            config={
                'agent_config': agent_config,
                'scenario_config': scenario_config,
                'rl_config': self.rl_config,
            },
        )
        
        king_generate_only = (
            self.mode == 'train_scenario'
            and scenario_config.get('policy_type') == 'king'
            and bool(scenario_config.get('king_train_scenario_only_generate', True))
        )

        # prepare parameters
        if self.mode == 'train_agent':
            self.buffer_capacity = agent_config['buffer_capacity']
            self.eval_in_train_freq = agent_config['eval_in_train_freq']
            self.save_freq = agent_config['save_freq']
            self.train_episode = agent_config['train_episode']
            # Follow ChatScene behavior.
            self.current_episode = -1
            self.logger.save_config(agent_config)
            self.logger.create_training_dir()
        elif self.mode == 'train_scenario' and not king_generate_only:
            self.buffer_capacity = scenario_config['buffer_capacity']
            self.eval_in_train_freq = scenario_config['eval_in_train_freq']
            self.save_freq = scenario_config['save_freq']
            self.train_episode = scenario_config['train_episode']
            self.logger.save_config(scenario_config)
            self.logger.create_training_dir()
        elif self.mode == 'train_scenario':
            # KING generation-only mode does not need replay buffer/training loop params.
            self.buffer_capacity = 0
            self.eval_in_train_freq = 0
            self.save_freq = scenario_config.get('save_freq', 1)
            self.train_episode = 0
            self.logger.save_config(scenario_config)
            self.logger.create_training_dir()
        elif self.mode == 'eval':
            self.save_freq = scenario_config['save_freq']
            self.logger.log('>> Evaluation Mode, skip config saving', 'yellow')
            self.logger.create_eval_dir(load_existing_results=True)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

        # define agent and scenario
        self.logger.log('>> Agent Policy: ' + agent_config['policy_type'])
        self.logger.log('>> Scenario Policy: ' + scenario_config['policy_type'])

        if self.scenario_config['auto_ego']:
            self.logger.log('>> Using auto-polit for ego vehicle, action of policy will be ignored', 'yellow')
        if scenario_config['policy_type'] == 'ordinary' and self.mode != 'train_agent':
            self.logger.log('>> Ordinary scenario can only be used in agent training', 'red')
            raise Exception()
        self.logger.log('>> ' + '-' * 40)

        # define agent and scenario policy
        self.agent_policy = AGENT_POLICY_LIST[agent_config['policy_type']](agent_config, logger=self.logger)
        self.scenario_policy = SCENARIO_POLICY_LIST[scenario_config['policy_type']](scenario_config, logger=self.logger)
        if self.save_video:
            assert self.mode == 'eval', "only allow video saving in eval mode"
            self.logger.init_video_recorder()

    def _init_world(self, town):
        self.logger.log(f">> Initializing carla world: {town}")
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.scenario_config['tm_port'])
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _init_renderer(self):
        self.logger.log(">> Initializing pygame birdeye renderer")
        pygame.init()
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN
        if self.scenario_category == 'planning': 
            # [bird-eye view, Lidar, front view] or [bird-eye view, front view]
            if self.env_params['disable_lidar']:
                window_size = (self.env_params['display_size'] * 2, self.env_params['display_size'] * self.num_scenario)
            else:
                window_size = (self.env_params['display_size'] * 3, self.env_params['display_size'] * self.num_scenario)
        else:
            window_size = (self.env_params['display_size'], self.env_params['display_size'] * self.num_scenario)
        self.display = pygame.display.set_mode(window_size, flag)

        # initialize the render for generating observation and visualization
        pixels_per_meter = self.env_params['display_size'] / self.env_params['obs_range']
        pixels_ahead_vehicle = (self.env_params['obs_range'] / 2 - self.env_params['d_behind']) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.env_params['display_size'], self.env_params['display_size']],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
        }
        self.birdeye_render = BirdeyeRender(self.world, self.birdeye_params, logger=self.logger)

    def train(self, data_loader, start_episode=0):
        # general buffer for both agent and scenario
        Buffer = RouteReplayBuffer if self.scenario_category == 'planning' else PerceptionReplayBuffer
        replay_buffer = Buffer(self.num_scenario, self.mode, self.buffer_capacity)
        env_steps_run = 0
        no_collision_episode_count = 0

        def route_avsafe_started():
            if not (av_safe._enabled and av_safe._training):
                return False
            if av_safe.n_updates() > 0:
                return True
            if self.env is None:
                return False
            return any(getattr(env, '_av_training_started', False) for env in self.env.env_list)

        for e_i in tqdm(range(start_episode, self.train_episode)):
            if av_safe._enabled and av_safe._training and av_safe.ready():
                self.logger.log('>> AV-safe converged on current route, stop training loop.', 'yellow')
                break
            if self.train_avsafe_steps is not None and av_safe.n_updates() >= self.train_avsafe_steps:
                self.logger.log(
                    f'>> Reached train_avsafe_steps={self.train_avsafe_steps}, stop training loop.',
                    'yellow'
                )
                break
            if (
                self.max_wait_collision_episodes is not None
                and av_safe._enabled and av_safe._training
                and not route_avsafe_started()
                and no_collision_episode_count >= self.max_wait_collision_episodes
            ):
                self.logger.log(
                    f'>> No collision observed in first {self.max_wait_collision_episodes} episodes; skip current route.',
                    'yellow'
                )
                break
            if self.train_env_steps is not None and env_steps_run >= self.train_env_steps:
                self.logger.log(f'>> Reached train_env_steps={self.train_env_steps}, stop training loop.', 'yellow')
                break
            # sample scenarios
            sampled_scenario_configs, _ = data_loader.sampler()
            # reset the index counter to create endless loader
            data_loader.reset_idx_counter()

            # get static obs and then reset with init action 
            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            scenario_init_action, additional_dict = self.scenario_policy.get_init_action(static_obs)
            obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
            replay_buffer.store_init([static_obs, scenario_init_action], additional_dict=additional_dict)

            # get ego vehicle from scenario
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            # # Sample one episode-level w for each parallel scenario, for CAPQL.
            # B = len(obs)  # Equals self.num_scenario
            # ep_w = self.scenario_policy.sample_pref(B)           # torch [B,K] on CUDA
            # ep_w_np = ep_w.detach().cpu().numpy().astype(np.float32)
            # self.logger.log(f"[PREF] episode-level w = {ep_w_np.tolist()}")

             # ===================== ScenePilot additions =====================
            # Episode-level sums: J_risk, J_sigma
            ep_sum_risk  = np.zeros(self.num_scenario, dtype=np.float64)
            ep_sum_sigma = np.zeros(self.num_scenario, dtype=np.float64)
            ep_stats = []  # Record each finished env {'J_risk':..., 'J_sigma':...}



            # start loop
            episode_reward = []
            # Read the currently active threshold; it is fixed until the next train() call.
            sigma_threshold_fn = getattr(self.scenario_policy, 'sigma_threshold_value', None)
            if callable(sigma_threshold_fn):
                th = sigma_threshold_fn()
                self.logger.log(f"[PREF] episode-level threshold = {th:.4f}")
            # Set before the episode starts.
            last_sigma = [None] * self.num_scenario
            stop_requested = False
            episode_collision_seen = False
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)

                # single objective version

                # ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                # scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)

                # # apply action to env and get obs
                # next_obs, rewards, dones, infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)
                # replay_buffer.store([ego_actions, scenario_actions, obs, next_obs, rewards, dones], additional_dict=infos)

                # muti-objective
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)

                scenario_actions = self.scenario_policy.get_action(
                    obs, infos, deterministic=False
                )

                # If env gates steer, prefer the actual executed action returned by env.
                next_obs, rewards, dones, infos = self.env.step(
                    ego_actions=ego_actions, scenario_actions=scenario_actions
                )
                episode_collision_seen = episode_collision_seen or any(bool(i.get('collision', 0)) for i in infos)

                # Each step.
                vec = np.stack([i['npc_moreward_vec'] for i in infos])  # [B,2]=[risk, sigma]
                curr_sigma = vec[:, 1].astype(float)
                for i in range(len(infos)):
                    # σ_t:Previous frame; use the current value as fallback for the first frame.
                    infos[i]['sigma_t']   = float(curr_sigma[i] if last_sigma[i] is None else last_sigma[i])
                    # σ_{t+1}:Current frame
                    infos[i]['sigma_tp1'] = float(curr_sigma[i])
                last_sigma = curr_sigma

                #Write the step's w into infos so it reaches the replay buffer. 
                # pw = ep_w_np  # Keep w fixed for each env.
                # for i in range(len(infos)):
                #     infos[i]['pref_w'] = pw[i]

                replay_buffer.store([ego_actions, scenario_actions, obs, next_obs, rewards, dones],
                                    additional_dict=infos)
                

                # ===================== ScenePilot additions =====================
                # Read vector rewards from infos and accumulate episode-level sums.
                # CarlaEnv._get_info() already writes 'npc_moreward_vec'.
                vec = np.stack([i['npc_moreward_vec'] for i in infos])  # [B, 2] = [risk, sigma]
                ep_sum_risk  += vec[:, 0]
                ep_sum_sigma += vec[:, 1]

                # For envs finishing on this tick, collect and reset their sums.
                for idx, d in enumerate(dones):
                    if d:
                        ep_stats.append({
                            'J_risk' : float(ep_sum_risk[idx]),
                            'J_sigma': float(ep_sum_sigma[idx]),
                        })
                        ep_sum_risk[idx] = 0.0
                        ep_sum_sigma[idx] = 0.0
                
                obs = copy.deepcopy(next_obs)
                episode_reward.append(np.mean(rewards))
                step_incr = len(rewards)
                env_steps_run += step_incr
                self.global_env_step += step_incr

                if av_safe._enabled and av_safe._training:
                    avsafe_stats = av_safe.latest_stats()
                    avsafe_update_step = avsafe_stats.get('avsafe_update_step')
                    if avsafe_update_step is not None and avsafe_update_step > self._last_avsafe_logged_update:
                        self._last_avsafe_logged_update = avsafe_update_step
                        route_avsafe_step = int(avsafe_update_step)
                        global_avsafe_step = self.avsafe_step_offset + route_avsafe_step
                        reward_arr = np.asarray(rewards)
                        reward_metrics = {
                            'scenario/reward_mean': float(np.mean(reward_arr)),
                        }
                        if reward_arr.ndim == 2 and reward_arr.shape[1] >= 2:
                            reward_metrics.update({
                                'scenario/reward_primary_mean': float(np.mean(reward_arr[:, 0])),
                                'scenario/reward_sigma_mean': float(np.mean(reward_arr[:, 1])),
                            })
                        self.logger.log_wandb(
                            {
                                'train/global_env_step': self.global_env_step,
                                'train/route_env_step': env_steps_run,
                                'train/global_avsafe_step': global_avsafe_step,
                                'train/route_avsafe_step': route_avsafe_step,
                                'train/episode': e_i,
                                'train/scenario_id': float(sampled_scenario_configs[0].scenario_id) if sampled_scenario_configs else None,
                                'train/route_id': float(sampled_scenario_configs[0].route_id) if sampled_scenario_configs else None,
                                'avsafe/loss': avsafe_stats.get('avsafe_loss'),
                                'avsafe/pred': avsafe_stats.get('avsafe_pred'),
                                'avsafe/tgt': avsafe_stats.get('avsafe_tgt'),
                                'avsafe/tgt_risk': avsafe_stats.get('avsafe_tgt_risk'),
                                'avsafe/td_reward': avsafe_stats.get('avsafe_td_reward'),
                                'avsafe/update_step': avsafe_stats.get('avsafe_update_step'),
                                'avsafe/weight': avsafe_stats.get('avsafe_weight'),
                                **reward_metrics,
                            },
                            step=global_avsafe_step,
                        )

                # ----- log av-safe every 500 updates -----
                if av_safe._training and av_safe.n_updates() % 500 == 0 and len(av_safe.loss_hist) and not av_safe._converged:
                    avsafe_stats = av_safe.latest_stats()
                    self.logger.add_training_results('global_env_step', self.global_env_step)
                    self.logger.add_training_results('global_avsafe_step', self.avsafe_step_offset + av_safe.n_updates())
                    self.logger.add_training_results('episode', e_i)
                    self.logger.add_training_results('risk_pred', avsafe_stats.get('avsafe_pred', av_safe.risk_hist[-1]))
                    self.logger.add_training_results('risk_loss', avsafe_stats.get('avsafe_loss', av_safe.loss_hist[-1]))
                    self.logger.add_training_results('risk_reward', avsafe_stats.get('avsafe_td_reward'))
                    self.logger.save_training_results()

                if self.train_avsafe_steps is not None and av_safe.n_updates() >= self.train_avsafe_steps:
                    self.logger.log(
                        f'>> Reached train_avsafe_steps={self.train_avsafe_steps} at global_avsafe_step={self.avsafe_step_offset + av_safe.n_updates()}.',
                        'yellow'
                    )
                    stop_requested = True

                if av_safe._enabled and av_safe._training and av_safe.ready():
                    self.logger.log(
                        f'>> AV-safe converged at global_avsafe_step={self.avsafe_step_offset + av_safe.n_updates()}.',
                        'yellow'
                    )
                    stop_requested = True

                if self.train_env_steps is not None and env_steps_run >= self.train_env_steps:
                    self.logger.log(
                        f'>> Reached train_env_steps={self.train_env_steps} at global_env_step={self.global_env_step}.',
                        'yellow'
                    )
                    stop_requested = True


                # train off-policy agent or scenario
                if self.mode == 'train_agent' and self.agent_policy.type == 'offpolicy':
                    self.agent_policy.train(replay_buffer)
                elif self.mode == 'train_scenario' and self.scenario_policy.type == 'offpolicy':
                    # Includes AV-safe handling after each tick.
                    self.scenario_policy.train(replay_buffer)

                if stop_requested:
                    break

            # end up environment
            self.env.clean_up()
            replay_buffer.finish_one_episode()
            self.logger.add_training_results('episode', e_i)
            self.logger.add_training_results('global_env_step', self.global_env_step)
            self.logger.add_training_results('global_avsafe_step', self.avsafe_step_offset + av_safe.n_updates())
            self.logger.add_training_results('episode_reward', np.sum(episode_reward))
            self.logger.save_training_results()

            if av_safe._enabled and av_safe._training and not route_avsafe_started() and not episode_collision_seen:
                no_collision_episode_count += 1
                self.logger.log(
                    f'>> No collision yet on current route: {no_collision_episode_count}/{self.max_wait_collision_episodes or "inf"} episodes.',
                    'yellow'
                )
                if self.max_wait_collision_episodes is not None and no_collision_episode_count >= self.max_wait_collision_episodes:
                    stop_requested = True
            elif route_avsafe_started():
                no_collision_episode_count = 0

            # train on-policy agent or scenario
            if self.mode == 'train_agent' and self.agent_policy.type == 'onpolicy':
                self.agent_policy.train(replay_buffer)
            elif self.mode == 'train_scenario' and self.scenario_policy.type in ['init_state', 'onpolicy']:
                # self.scenario_policy.train(replay_buffer)
                 # Pass the rollout stats here and log k changes during training; sampling above still uses the previously active k.
                self.scenario_policy.train(replay_buffer, ep_stats=ep_stats)

            # eval during training
            if (e_i+1) % self.eval_in_train_freq == 0:
                #self.eval(env, data_loader)
                pass
            
            # if len(av_safe.loss_hist) >= 1:
            #     self.logger.add_training_results('risk_loss', av_safe.loss_hist[-1])
            #     self.logger.add_training_results('risk_pred', av_safe.risk_hist[-1])
            # save checkpoints
            if (e_i+1) % self.save_freq == 0:
                if self.mode == 'train_agent':
                    self.agent_policy.save_model(e_i)
                if self.mode == 'train_scenario':
                    self.scenario_policy.save_model(e_i)
            if av_safe._enabled and av_safe._training and not av_safe.converged():
                ckpt_path = self.rl_config['av_safe']['ckpt']
                av_safe.save(ckpt_path)
                self.logger.log(f'>> Saved AV-safe to {ckpt_path}')

            if stop_requested:
                break
            
            # # Online AV-safe fine-tuning may be added after stabilization.
            # if av_safe._enabled and (e_i+1) % self.save_freq == 0:
            #     ckpt_path = self.rl_config['av_safe']['ckpt']
            #     av_safe.save(ckpt_path)
            #     self.logger.log(f'>> Saved AV-safe to {ckpt_path}')


    # ==========================================================
    #  ChatScene Agent training: episode loop that trains only the agent.
    # ==========================================================
    def train_agent(self, data_loader, start_episode=0, replay_buffer=None):
        """
        - data_loader: only provides one ScenarioConfig from ScenicDataLoader or ScenarioDataLoader
        - replay_buffer: globally reused Route/Perception buffer
        """

        from tqdm import tqdm
        for _ in tqdm(range(len(data_loader))):
            self.current_episode += 1
            if self.current_episode >= self.train_episode:
                return
            if self.current_episode < start_episode:
                continue

            # 1) Get one scenario config.
            sampled_cfgs, _ = data_loader.sampler()

            self.scenario_policy.set_mode('eval')
            self.scenario_policy.load_model(sampled_cfgs)
            

            # 2) Reset env without scenic.run_scenes.
            static_obs = self.env.get_static_obs(sampled_cfgs)
            scenario_init_act, add_dict = self.scenario_policy.get_init_action(static_obs)
            obs, infos = self.env.reset(sampled_cfgs, scenario_init_act)
            replay_buffer.store_init([static_obs, scenario_init_act], additional_dict=add_dict)

            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            # 3) rollout
            ep_reward = []
            while not self.env.all_scenario_done():
                ego_a = self.agent_policy.get_action(obs, infos, deterministic=False)
                sc_a  = self.scenario_policy.get_action(obs, infos, deterministic=False)
                nxt_obs, rewards, _, infos = self.env.step(ego_a, sc_a)

                replay_buffer.store([ego_a, sc_a, obs, nxt_obs, rewards, _],
                                    additional_dict=infos)
                obs = nxt_obs
                ep_reward.append(np.mean(rewards))

            # 4) **One-shot** off-policy update
            if self.agent_policy.type == 'offpolicy':
                loss = self.agent_policy.train(replay_buffer)

            score_function = get_route_scores if self.scenario_category in ['planning', 'scenic'] else get_perception_scores
            all_scores = score_function(self.env.running_results)


            # 5) End-of-episode cleanup and logging
            self.env.clean_up()
            replay_buffer.finish_one_episode()
            self.logger.add_training_results('episode', self.current_episode)
            self.logger.add_training_results('episode_reward', np.sum(ep_reward))
            for key, value in all_scores.items():
                self.logger.add_training_results(key, value)

            if loss is not None:
                critic_loss, actor_loss = loss
                self.logger.add_training_results('critic_loss', critic_loss)
                self.logger.add_training_results('actor_loss', actor_loss)
            else:
                critic_loss, actor_loss = 0, 0
                self.logger.add_training_results('critic_loss', critic_loss)
                self.logger.add_training_results('actor_loss', actor_loss)  
            self.logger.log(f">> Episode: {self.current_episode}, #buffer_len: {replay_buffer.buffer_len}, critic: {critic_loss:.3f}, actor: {actor_loss:.3f}")
            self.logger.save_training_results()

            # 6) On-policy update
            if self.agent_policy.type == 'onpolicy':
                self.agent_policy.train(replay_buffer)

            # 7) Periodic save
            if (self.current_episode + 1) % self.save_freq == 0:
                self.agent_policy.save_model(self.current_episode, replay_buffer)


    def eval(self, data_loader):
        num_finished_scenario = 0
        data_loader.reset_idx_counter()
        while len(data_loader) > 0:
            # sample scenarios
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            num_finished_scenario += num_sampled_scenario

            # reset envs with new config, get init action from scenario policy, and run scenario
            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            self.scenario_policy.load_model(sampled_scenario_configs)
            scenario_init_action, _ = self.scenario_policy.get_init_action(static_obs, deterministic=True)
            obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)

            # get ego vehicle from scenario
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            score_list = {s_i: [] for s_i in range(num_sampled_scenario)}
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=True)
                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=True)

                # apply action to env and get obs
                # This makes all learning algorithms use the same reward.
                obs, rewards, _, infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)

                # save video
                if self.save_video:
                    if self.scenario_category == 'planning':
                        self.logger.add_frame(pygame.surfarray.array3d(self.display).transpose(1, 0, 2))
                    else:
                        self.logger.add_frame({s_i['scenario_id']: ego_actions[n_i]['annotated_image'] for n_i, s_i in enumerate(infos)})

                # accumulate scores of corresponding scenario
                reward_idx = 0
                for s_i in infos:
                    score = rewards[reward_idx] if self.scenario_category == 'planning' else 1-infos[reward_idx]['iou_loss']
                    score_list[s_i['scenario_id']].append(score)
                    reward_idx += 1

            # clean up all things
            self.logger.log(">> All scenarios are completed. Clearning up all actors")
            self.env.clean_up()

            # save video
            if self.save_video:
                data_ids = [config.data_id for config in sampled_scenario_configs]
                self.logger.save_video(data_ids=data_ids)

            # print score for ranking
            self.logger.log(f'[{num_finished_scenario}/{data_loader.num_total_scenario}] Ranking scores for batch scenario:', 'yellow')
            for s_i in score_list.keys():
                self.logger.log('\t Env id ' + str(s_i) + ': ' + str(np.mean(score_list[s_i])), 'yellow')

            # calculate evaluation results
            score_function = get_route_scores if self.scenario_category == 'planning' else get_perception_scores
            all_running_results = self.logger.add_eval_results(records=self.env.running_results)
            all_scores = score_function(all_running_results)
            self.logger.add_eval_results(scores=all_scores)
            self.logger.print_eval_results()
            if len(self.env.running_results) % self.save_freq == 0:
                self.logger.save_eval_results()
        self.logger.save_eval_results()

    def run(self):
        # get scenario data of different maps
        config_by_map = scenario_parse(self.scenario_config, self.logger)
        if (
            self.mode == 'train_scenario'
            and self.scenario_config.get('policy_type') == 'king'
            and bool(self.scenario_config.get('king_train_scenario_only_generate', True))
        ):
            total = sum(len(v) for v in config_by_map.values())
            self.logger.log(f'>> KING train_scenario finished generation/indexing ({total} scenarios), skip rollout.')
            return

        # 2) For train_agent, repeat each town's configs 20 times.
        if self.mode == 'train_agent':
            config_by_map = {
                # it may casue some maps don't run, becuase *20 if maps have 3 routes 30*20=600, it don't run other maps
                # town: cfgs * 20
                town: cfgs * 15
                for town, cfgs in config_by_map.items()
            }

        for m_i in config_by_map.keys():
            # initialize map and render
            self._init_world(m_i)
            self._init_renderer()

            # create scenarios within the vectorized wrapper
            self.env = VectorWrapper(
                self.env_params,  # Includes rl_safe_cfg
                self.scenario_config, 
                self.world, 
                self.birdeye_render, 
                self.display, 
                self.logger
            )

            # prepare data loader and buffer
            data_loader = ScenarioDataLoader(config_by_map[m_i], self.num_scenario, m_i, self.world)


            # run with different modes
            if self.mode == 'eval':
                self.agent_policy.load_model()
                # self.scenario_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('eval')
                self.eval(data_loader)

            elif self.mode == 'train_agent':
                # Original train_agent path
                # start_episode = self.check_continue_training(self.agent_policy)
                # self.scenario_policy.load_model()
                # self.agent_policy.set_mode('train')
                # self.scenario_policy.set_mode('eval')
                # self.train(data_loader, start_episode)

                #ChatScene agent-training path 
                # -------- 0) Build global buffer --------
                BufferCls = RouteReplayBuffer if self.scenario_category == 'planning' else PerceptionReplayBuffer
                replay_buf = BufferCls(self.num_scenario, 'train_agent', self.buffer_capacity, mix_collision=True)

                # -------- 2) Resume training --------,Be careful not to delete previous weights when switching scenarios.                         
                # if self.continue_agent_training:
                #     self.logger.load_training_results()
                #     start_ep = self.check_continue_training(self.agent_policy, replay_buffer=replay_buf) + 1
                #     if start_ep >= self.train_episode:
                #         return
                # else:
                #     self.clean_cache(self.agent_policy.model_path)
                #     start_ep = -1

        
                self.agent_policy.set_mode('train')

                # ------------ Run one real episode in ChatScene style ------------
                self.train_agent(data_loader, start_episode=0, replay_buffer=replay_buf)
                
            elif self.mode == 'train_scenario':
                start_episode = self.check_continue_training(self.scenario_policy)
                self.agent_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('train')
                self.train(data_loader, start_episode)
            else:
                raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    def check_continue_training(self, policy):
        # load previous checkpoint
        policy.load_model()
        if policy.continue_episode == 0:
            start_episode = 0
            self.logger.log('>> Previous checkpoint not found. Training from scratch.')
        else:
            start_episode = policy.continue_episode
            self.logger.log('>> Continue training from previous checkpoint.')
        return start_episode
    

    def clean_cache(self, path):
        # Get a list of all files in directory
        all_files = glob.glob(os.path.join(path, '*'))

        # Specify the file to keep
        file_to_keep = os.path.join(path, 'model.sac.-001.torch')

        # Remove all files except the one to keep
        for file in all_files:
            if file != file_to_keep:
                os.remove(file)
                

    def close(self):
        pygame.quit() 
        if self.env:
            self.env.clean_up()
