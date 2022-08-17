import gym
import argparse

from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
import d4rl
from hrl.ope.utils import load_chain

parser = argparse.ArgumentParser()
parser.add_argument("--base_fname", type=str, required=True)
args = parser.parse_args()

chain = load_chain(args.base_fname)