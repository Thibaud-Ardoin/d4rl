import argparse
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, VAEPolicy
from rlkit.torch.sac.bear import BEARTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.environments import gym_wrapper

from flow.controllers import IDMController, ContinuousRouter, FollowerStopper, RandomController, RLController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs.ring.accel import AccelEnv
from flow.envs.ring.wave_attenuation import WaveAttenuationPOEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS


from torch import autograd

import h5py, argparse, os
import ray
import gym
import d4rl

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu

import numpy as np
import argparse
import gym
import d4rl.flow
from d4rl.utils import dataset_utils

from flow.controllers import car_following_models

file='/home/tibo/Documents/Prog/Git/d4rl/d4rl_evaluations/bear/data/BEAR-launch/27189/BEAR_launch/27189_2020_08_11_13_10_11_0000--s-0/params.pkl'
env_name='flow-ring-v0'


# Load network Data
network_data = torch.load(file)
policy = network_data['trainer/policy']
qf1 = network_data['trainer/qf1']
qf2 = network_data['trainer/qf2']
target_qf1 = network_data['trainer/target_qf1']
target_qf2 = network_data['trainer/target_qf2']
vae = network_data['trainer/vae']

eval_env = gym.make(env_name)
expl_env = eval_env

eval_path_collector = CustomMDPPathCollector(
    eval_env,
)
expl_path_collector = MdpPathCollector(
    expl_env,
    policy,
)
replay_buffer_size=int(2E4),
buffer_filename = None

print(replay_buffer_size)
replay_buffer = EnvReplayBuffer(
    replay_buffer_size[0],
    expl_env,
)
# load_hdf5(offline_dataset, replay_buffer, max_size=replay_buffer_size)

algorithm_kwargs=dict(
    num_epochs=100,
    num_eval_steps_per_epoch=100,
    num_trains_per_train_loop=100,
    num_expl_steps_per_train_loop=100,
    min_num_steps_before_training=100,
    max_path_length=1000,
    batch_size=256,
    num_actions_sample=100,
)

trainer = BEARTrainer(
    env=eval_env,
    policy=policy,
    qf1=qf1,
    qf2=qf2,
    target_qf1=target_qf1,
    target_qf2=target_qf2,
    vae=vae
)
algorithm = TorchBatchRLAlgorithm(
    trainer=trainer,
    exploration_env=expl_env,
    evaluation_env=eval_env,
    exploration_data_collector=expl_path_collector,
    evaluation_data_collector=eval_path_collector,
    replay_buffer=replay_buffer,
    batch_rl=True,
    q_learning_alg=True,
    **algorithm_kwargs
)


from flow.controllers.base_controller import BaseController


class RLTestConntroller(BaseController):

    def __init__(self, veh_id, car_following_params):
        """Instantiate an RL Controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params)

    def get_accel(self, env):
        action = algorithm.policy_fn(env.states)
        print('Action: ', action)
        return action



vehicles = VehicleParams()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {'noise': 0}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=21)

vehicles.add(
    veh_id="rl",
    # acceleration_controller=(IDMController, {'noise': 1}),
    acceleration_controller=(RLTestConntroller, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1)


flow_params = dict(
    # name of the experiment
    exp_tag='ring',

    rl_actions=rl_actions,

    # name of the flow environment the experiment is running on
    env_name=WaveAttenuationPOEnv, #AccelEnv

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.1,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=3000,
        additional_params=ADDITIONAL_ENV_PARAMS,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        bunching=20,
    ),
)
