import os
from argparse import ArgumentParser
import engine
import json
import datetime as dt


def dt2str(time_):
    return f'{time_:%Y-%m-%d_%H-%M-%S}'



def main(args):
    with open(args.conf, 'r') as jin:
        config = json.load(jin)
    config['time'] = dt.datetime.now()
    # the model weight should save at this place.
    config['logs']['exp_dir'] = os.path.join(config['logs']['exp_root'], dt2str(config['time']))
    engine.start_training(config)
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--conf', default='./conf/train_unet.json')
    # parser.add_argument('--world-size', default=-1)
    # parser.add_argument('--rank', default=-1)
    # parser.add_argument('--dist-url', default='tcp://localhost:23456')
    # parser.add_argument('--dist-backend', default='nccl')
    # parser.add_argument('--local_rank')
    main(parser.parse_args())
