# ScenePilot: Controllable Boundary-Driven Critical Scenario Generation for Autonomous Driving

ScenePilot is a CARLA-based safety-critical autonomous driving project built on top of the SafeBench codebase. It trains and evaluates adversarial scenario policies that generate challenging traffic interactions for an ego autonomous vehicle. The project adds ScenePilot scenario-policy training, AV-safe risk estimation, physical-safety reward shaping, multi-route training scripts, and evaluation utilities.


## Project Structure

```text
ScenePilot/
├── README.md
├── environment.yml                  # Conda environment specification
├── scripts/                         # Run, train, evaluate, and CARLA helper scripts
├── safebench/
│   ├── agent/                       # Ego-agent policies and configs
│   ├── gym_carla/                   # CARLA environment wrapper, rewards, replay buffer
│   ├── scenario/                    # Scenario policies, definitions, data loaders, configs
│   └── util/                        # Logging, metrics, torch helpers
├── tools/                           # Scenario/route generation tools
├── docs/                            # Documentation inherited from SafeBench
├── console/                         # Runtime console logs
├── log_scenario/                    # Training/evaluation outputs
└── train_logs/                      # Batch training logs
```

## Local Installation

### Step 1: Create the conda environment

```bash
conda env create -f environment.yml
conda activate scenepilot
```

The project targets Python 3.8, CARLA 0.9.13, and PyTorch with CUDA 11.8.

### Step 2: Enter the project root

```bash
cd ScenePilot
export PYTHONPATH="$PWD:$PYTHONPATH"
```

There is no `setup.py` in this copy of the project, so `PYTHONPATH` is the recommended way to make the `safebench` package importable.

### Step 3: Install system dependencies for CARLA

```bash
sudo apt install libomp5
```

This is commonly required by CARLA on Linux.

### Step 4: Download CARLA 0.9.13

Download [CARLA 0.9.13_safebench](https://drive.google.com/file/d/139vLRgXP90Zk6Q_du9cRdOLx7GJIw_0v/view) and extract it to a local folder.


### Step 5: Add CARLA Python API paths

Add the following lines to `~/.bashrc` or run them in every terminal before starting experiments:

```bash
export CARLA_ROOT=~/CARLA/CARLA_0.9.13_safebench
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
```

Reload the shell:

```bash
source ~/.bashrc
```

## CARLA Setup

### Desktop Users

```bash
cd "$CARLA_ROOT"
./CarlaUE4.sh -prefernvidia -windowed -carla-port=3000
```

### Remote Server Users

```bash
cd "$CARLA_ROOT"
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=3000
```

### Multiple CARLA Servers

```bash
cd ScenePilot
bash scripts/start_carla.sh -n 2 -b 3000 -s 4 -C "$CARLA_ROOT"
```

This starts CARLA on ports `3000`, `3004`, and so on. Match the training script ports with the CARLA ports.

## Important Paths

- Agent configs: `safebench/agent/config/`
- Scenario configs: `safebench/scenario/config/`
- ScenePilot config: `safebench/scenario/config/scenepilot.yaml`
- Reward and AV-safe config: `scripts/rlconfig.yaml`
- ScenePilot checkpoints: `safebench/scenario/scenario_data/model_ckpt/scenepilot/`

Routes `4-13` are CARLA maps. Routes `0-3` are SafeBench own maps.

When changing the scenario id, check `safebench/scenario/config/scenepilot.yaml`. Some scenarios use different `scenario_state_dim` and `scenario_action_dim` values.

## Supported Ego-Agent Configs

The ego-agent configuration is selected with `--agent_cfg`. Common options are:

- `autopilot.yaml`: Expert autopilot baseline using CARLA Traffic Manager.
- `chatscene.yaml`: RL ego-agent configuration for SAC, PPO, and TD3 baselines, using pretrained checkpoints from ChatScene.
- `behavior.yaml`: Rule-based ego-agent baseline using CARLA's `BehaviorAgent`.
- `aim_bev.yaml`: AIM-BEV policy config using the checkpoint under `safebench/agent/model_ckpt/aim_bev/regular`.
- `transfuser.yaml`: TransFuser policy config using camera, lidar, and state observations.

Make sure the checkpoint paths inside the selected YAML file exist before launching training or evaluation.

## Train AV-Safe with PPO

AV-safe training is launched with `scripts/start_training_avsafe_ppo.sh`. The script runs route-level AV-safe training sequentially across the configured scenario and route list.

Start one CARLA server first, then run:

```bash
cd ScenePilot
bash scripts/start_training_avsafe_ppo.sh \
  --device cuda:0 \
  --port 3000 \
  --tm_port 9000 \
  --agent_cfg chatscene.yaml \
  --scenario_cfg ppo.yaml
```

Useful overrides:

```text
--avsafe_steps_per_route N        AV-safe update steps per route
--max_wait_collision_episodes N   Skip a route after N collision-free episodes
--exp_name NAME                   Experiment name for logs/checkpoints
--wandb_mode online|offline|disabled
```

## Run a Single ScenePilot Training Job

Start CARLA first, then run:

```bash
cd ScenePilot
export PYTHONPATH="$PWD:$PYTHONPATH"

python scripts/run.py \
  --tag 6-4 \
  --mode train_scenario \
  --agent_cfg chatscene.yaml \
  --scenario_cfg scenepilot.yaml \
  --scenario_id 6 \
  --route_id 4 \
  --port 3000 \
  --tm_port 9000 \
  --device cuda:0
```

Useful options:

```text
--train_env_steps N              Stop after N environment steps
--train_avsafe_steps N           Stop after N AV-safe update steps
--avsafe_training 0|1            Override scripts/rlconfig.yaml av_safe.training
--use_wandb                      Enable Weights & Biases logging
--wandb_project PROJECT_NAME     W&B project name
```

## Batch Scenario Training

Launch consecutive routes for one scenario id:

```bash
cd ScenePilot
bash scripts/start_training_sce.sh \
  -S 6 \
  -r 4 \
  -k 10 \
  -p 3000 \
  -s 4 \
  --tm_base 9000 \
  --device cuda:0
```

This starts tags `6-4`, `6-5`, ..., with CARLA ports `3000`, `3004`, ... and Traffic Manager ports `9000`, `9004`, ...

Launch arbitrary scenario-route pairs:

```bash
bash scripts/start_training_sce_list.sh \
  --pairs "6-4 6-5 7-5" \
  -p 3000 \
  --tm_base 9000 \
  -s 4 \
  --devices "cuda:0,cuda:1"
```

The batch scripts resolve `REPO_ROOT` from their own location and use `python` from the active shell by default. Override Python explicitly if needed:

```bash
PYTHON=/path/to/conda/envs/scenepilot/bin/python bash scripts/start_training_sce.sh -S 6 -r 4
```

## Train an Ego Agent

The ego agent can be trained with `train_agent` mode:

```bash
cd ScenePilot
export PYTHONPATH="$PWD:$PYTHONPATH"

python scripts/run_train_av.py \
  --tag av-6 \
  --mode train_agent \
  --agent_cfg chatscene.yaml \
  --scenario_cfg scenepilot.yaml \
  --scenario_id 6 \
  --port 3000 \
  --tm_port 9000 \
  --device cuda:0
```

Helper script:

```bash
bash scripts/start_training_av.sh
```

## Scenario Evaluation

Start CARLA first, then run:

```bash
cd ScenePilot
export PYTHONPATH="$PWD:$PYTHONPATH"

python scripts/run_eval.py \
  --tag eval-6-4 \
  --agent_cfg eval_gen.yaml \
  --scenario_cfg scenepilot.yaml \
  --scenario_id 6 \
  --route_id 4 \
  --port 3000 \
  --tm_port 9000 \
  --device cuda:0
```

Batch scenario evaluation:

```bash
bash scripts/start_eval_batch.sh \
  -s "6 7 8" \
  -a transfuser.yaml \
  -c king.yaml \
  -p 3000 \
  -M 9000
```

Everything after `--` is forwarded to `scripts/run_eval.py`, for example:

```bash
bash scripts/start_eval_batch.sh -s "6" -a chatscene.yaml -c scenepilot.yaml -- --route_id 4
```

## Finetuned Ego-Agent Evaluation

Use `scripts/run_eval_av.py` to evaluate a finetuned ego-agent checkpoint directly:

```bash
cd ScenePilot
export PYTHONPATH="$PWD:$PYTHONPATH"

python scripts/run_eval_av.py \
  --tag eval-av-7 \
  --agent_cfg chatscene.yaml \
  --scenario_cfg scenepilot.yaml \
  --scenario_id 7 \
  --route_id 4 \
  --load_dir safebench/agent/model_ckpt/sac_chatscene/finetune/7 \
  --load_iteration 100 \
  --port 3000 \
  --tm_port 9000 \
  --device cuda:0
```

To evaluate all checkpoints under a finetune directory, use the batch helper:

```bash
bash scripts/start_batch_eval_av.sh
```

Override the defaults:

```bash
PARALLEL=2 \
SCENARIO_IDS="7 8" \
BASE_PORT=3000 \
BASE_TM_PORT=9000 \
bash scripts/start_batch_eval_av.sh \
  safebench/agent/model_ckpt/sac_chatscene/finetune \
  chatscene.yaml \
  scenepilot.yaml
```

## Modes

- `train_scenario`: trains the adversarial scenario policy. This is the main ScenePilot mode.
- `train_agent`: trains the ego-agent policy under generated scenarios.
- `eval`: evaluates a trained ego agent and scenario setup.

## Checkpoints

Pretrained checkpoints and related model files:

[ScenePilot Checkpoints](https://drive.google.com/drive/folders/1bUP66-DBsQDn4XJciQS6Ug7oIxWim4bz?usp=drive_link)

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{
ruan2026scenepilot,
title={ScenePilot: Controllable Boundary-Driven Critical Scenario Generation for Autonomous Driving},
author={Qiyu Ruan and YUXUAN WANG and He Li and Zhenning Li and Cheng-zhong Xu},
booktitle={Forty-third International Conference on Machine Learning},
year={2026}
}
```

## Acknowledgement

This implementation is based on code and ideas from several repositories. We sincerely thank the authors for their work.

- [SafeBench](https://github.com/trust-ai/SafeBench)
- [ChatScene](https://github.com/javyduck/ChatScene)
- [KING](https://github.com/autonomousvision/king/tree/main)
- [TransFuser](https://github.com/autonomousvision/transfuser/tree/cvpr2021)
- [FREA](https://github.com/CurryChen77/FREA)
