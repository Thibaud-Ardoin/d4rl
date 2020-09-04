import numpy as np
import argparse
import gym
import d4rl.flow
from d4rl.utils import dataset_utils

from flow.controllers import car_following_models, FollowerStopper

"""
Generate controled (Or uncontroled) data from flow simulator
"""


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--render', action='store_true', help='Render trajectories')
    #parser.add_argument('--type', action='store_true', help='Noisy actions')
    parser.add_argument('--controller', type=str, default='idm', help='random, idm, follower-stopper')
    parser.add_argument('--env_name', type=str, default='flow-ring-v0', help='Maze type. small or default')
    parser.add_argument('--num_samples', type=int, default=int(1), help='Num samples to collect')
    parser.add_argument('--noise', type=str, default='1.0', help='Qtt of noise added to idm for data generation')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    print('env : ', env)
    env.reset()
    print(env.action_space)

    if args.controller == 'idm':
        uenv = env.unwrapped
        veh_ids = uenv.k.vehicle.get_rl_ids()
        if hasattr(uenv, 'num_rl'):
            num_rl = uenv.num_rl
        else:
            num_rl = len(veh_ids)
        if num_rl == 0:
            raise ValueError("No RL vehicles")
        controllers = []

        acc_controller = uenv.k.vehicle.get_acc_controller(uenv.k.vehicle.get_ids()[0])
        car_following_params = acc_controller.car_following_params
        #for veh_id in veh_ids:
        #    controllers.append(car_following_models.IDMController(veh_id, car_following_params=car_following_params))

        def get_action(s):
            actions = np.zeros_like(env.action_space.sample())
            for i, veh_id in enumerate(uenv.k.vehicle.get_rl_ids()):
                if i >= actions.shape[0]:
                    break
                actions[i] = car_following_models.IDMController(veh_id, car_following_params=car_following_params, noise=float(args.noise)).get_action(env)
            return actions

    elif args.controller == 'random':
        def get_action(s):
            return env.action_space.sample()

    elif args.controller == 'follower-stopper':
        uenv = env.unwrapped
        veh_ids = uenv.k.vehicle.get_rl_ids()
        if hasattr(uenv, 'num_rl'):
            num_rl = uenv.num_rl
        else:
            num_rl = len(veh_ids)
        if num_rl == 0:
            raise ValueError("No RL vehicles")
        controllers = []

        acc_controller = uenv.k.vehicle.get_acc_controller(uenv.k.vehicle.get_ids()[0])
        car_following_params = acc_controller.car_following_params
        #for veh_id in veh_ids:
        #    controllers.append(car_following_models.IDMController(veh_id, car_following_params=car_following_params))

        def get_action(s):
            actions = np.zeros_like(env.action_space.sample())
            for i, veh_id in enumerate(uenv.k.vehicle.get_rl_ids()):
                if i >= actions.shape[0]:
                    break
                actions[i] = FollowerStopper(veh_id, car_following_params=car_following_params, v_des=2.5).get_accel(env)
            return actions

    else:
        raise ValueError("Unknown controller type: %s" % str(args.controller))

    writer = dataset_utils.DatasetWriter()
    while len(writer) < args.num_samples:
        s = env.reset()
        ret = 0
        for _ in range(env._max_episode_steps):
            action = get_action(s)
            ns , r, done, infos = env.step(action)
            ret += r
            writer.append_data(s, action, r, done)
            s = ns
        print(ret, '; len=', len(writer), ' / ', args.num_samples)
        # env.render()
        noise = args.noise.replace('.', '+')
        fname = 'data_generation/%s-%s-noise%s.hdf5' % (args.env_name, args.controller, noise)
        writer.write_dataset(fname, max_size=args.num_samples)

if __name__ == "__main__":
    main()
