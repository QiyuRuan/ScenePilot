''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-22 18:45:01
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import traceback
import os.path as osp

import torch 

from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.carla_runner import CarlaRunner
import re


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, default=None, help='short run tag shown in logs/dirs')

    parser.add_argument('--exp_name', type=str, default='exp')
    # parser.add_argument('--output_dir', type=str, default='log')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))

    parser.add_argument('--max_episode_step', type=int, default=300)
    parser.add_argument('--auto_ego', action='store_true')
    parser.add_argument('--mode', '-m', type=str, default='train_agent', choices=['train_agent', 'train_scenario', 'eval'])
    parser.add_argument('--agent_cfg', nargs='*', type=str, default=['chatscene.yaml'])
    parser.add_argument('--scenario_cfg', nargs='*', type=str, default=['scenepilot.yaml'])
    parser.add_argument('--continue_agent_training', '-cat', type=bool, default=False)
    parser.add_argument('--continue_scenario_training', '-cst', type=bool, default=False)

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')   

    # Run multiple scenarios independently.
    parser.add_argument('--num_scenario', '-ns', type=int, default=1, help='num of scenarios we run in one episode')

    parser.add_argument('--save_video', action='store_true',default=False)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=3000, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=9000, help='traffic manager port')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)

    parser.add_argument('--route_id', '--route_id_spy', dest='route_id', type=int, default=None)
    parser.add_argument('--scenario_id', '--scenario_id_spy', dest='scenario_id', type=int, default=None)

    parser.add_argument('--scenario_ratio', type=str, default=None, help='override scenario_folder (ratio dir)')
    parser.add_argument('--scenario_model_root', type=str, default=None, help='override scenario model ckpt root')

    args = parser.parse_args()
    run_tag = args.tag
    if args.output_dir is None:
        args.output_dir = f"log_ft_AV/log_{run_tag}"
    args_dict = vars(args).copy()
    args_dict.pop('scenario_id', None)
    args_dict.pop('scenario_model_root', None)

         # ---------- tee stdout/stderr ----------
    import sys, datetime, os
    log_path = os.path.join("console","finetune_AV"
                            f'console_{run_tag}.txt')
    ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")   # Generic match
    os.makedirs("console", exist_ok=True)
    tee_f = open(log_path, 'a')
    class TeeStdout:
        def __init__(self, term, logfile):
            self.term, self.file = term, logfile
        def write(self, txt):
            self.term.write(txt)                       # Keep colors in the terminal.
            clean = ansi_re.sub("", txt)               # Strip colors.
            self.file.write(clean)
        def flush(self):
            self.term.flush(); self.file.flush()

    sys.stdout = TeeStdout(sys.__stdout__, tee_f)     # Replace stdout only.
    
    # Load configured device.
    _default_device = args.device
    _old_torch_load = torch.load
    def _patched_torch_load(f, *args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = _default_device
        return _old_torch_load(f, *args, **kwargs)
    torch.load = _patched_torch_load

    err_list = []
    for agent_cfg in args.agent_cfg:
        for scenario_cfg in args.scenario_cfg:
            # set global parameters
            set_torch_variable(args.device)
            torch.set_num_threads(args.threads)
            set_seed(args.seed)

            # load agent config
            agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', agent_cfg)
            agent_config = load_config(agent_config_path)

            # load scenario config
            scenario_config_path = osp.join(args.ROOT_DIR, 'safebench/scenario/config', scenario_cfg)
            scenario_config = load_config(scenario_config_path)

            # load RL_safe config
            rl_path = osp.join(args.ROOT_DIR, 'scripts/rlconfig.yaml')
            rl_config = load_config(rl_path)

            # main entry with a selected mode
            agent_config.update(args_dict)
            scenario_config.update(args_dict)
            # rl_safe_config.update(args_dict)
            scenario_config['rl_config'] = rl_config

            # override for muti trian scenario .sh
            if args.route_id is not None:
                scenario_config['route_id'] = args.route_id

            if args.scenario_id is not None:
                scenario_config['scenario_id'] = args.scenario_id

            # ---- overrides for train agent (AV) ----
            if args.scenario_ratio is not None:
                scenario_config['scenario_ratio'] = args.scenario_ratio

            scenario_model_root = args.scenario_model_root or scenario_config.get('model_path')
            policy_type = scenario_config.get('policy_type')
            scenario_id = scenario_config.get('scenario_id')

            if scenario_id is not None:
                if args.scenario_ratio is not None:
                    agent_config['load_dir'] = os.path.join(
                        agent_config['load_dir'], str(scenario_id), str(args.scenario_ratio)
                    )
                else:
                    agent_config['load_dir'] = os.path.join(agent_config['load_dir'], str(scenario_id))

            if scenario_model_root:
                if policy_type in {'scenepilot', 'scenepilot'}:
                    scenario_config['model_path'] = scenario_model_root
                else:
                    if scenario_id is not None and args.scenario_ratio is not None:
                        scenario_config['model_path'] = os.path.join(
                            scenario_model_root, str(scenario_id), str(args.scenario_ratio)
                        )
                    elif scenario_id is not None:
                        scenario_config['model_path'] = os.path.join(scenario_model_root, str(scenario_id))
                    else:
                        scenario_config['model_path'] = scenario_model_root
                
            

            if scenario_config['policy_type'] == 'scenic':
                from safebench.scenic_runner import ScenicRunner
                assert scenario_config['num_scenario'] == 1, 'the num_scenario can only be one for scenic now'
                runner = ScenicRunner(agent_config, scenario_config)
            else:
                runner = CarlaRunner(agent_config, scenario_config)

            # start running
            try:
                runner.run()
            except:
                runner.close()
                traceback.print_exc()
                err_list.append([agent_cfg, scenario_cfg, traceback.format_exc()])

    for err in err_list:
        print(err[0], err[1], 'failed!')
        print(err[2])
