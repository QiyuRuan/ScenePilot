''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 16:25:21
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

# collect policy models from scenarios
from safebench.scenario.scenario_policy.dummy_policy import DummyPolicy
from safebench.scenario.scenario_policy.reinforce_continuous import REINFORCE
from safebench.scenario.scenario_policy.normalizing_flow_policy import NormalizingFlow
from safebench.scenario.scenario_policy.hardcode_policy import HardCodePolicy
from safebench.scenario.scenario_policy.adv_patch import ObjectDetection
from safebench.scenario.scenario_policy.rl.sac import SAC
from safebench.scenario.scenario_policy.rl.ppo import PPO

from safebench.scenario.scenario_policy.rl.scenepilot import ScenePilot
from safebench.scenario.scenario_policy.king_policy import KingPolicy



SCENARIO_POLICY_LIST = {
    'standard': DummyPolicy,
    'ordinary': DummyPolicy,
    'scenic': DummyPolicy,
    'advsim': HardCodePolicy,
    'advtraj': HardCodePolicy,
    'human': HardCodePolicy,
    'random': HardCodePolicy,
    'lc': REINFORCE,
    'nf': NormalizingFlow,
    'od': ObjectDetection,
    'sac': SAC,
    'ppo': PPO,
    'scenepilot': ScenePilot,
    'king': KingPolicy,
}

