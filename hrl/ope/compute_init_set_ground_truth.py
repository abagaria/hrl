from hrl.ope.utils import load_chain

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_fname", type=str, required=True)
args = parser.parse_args()

chain = load_chain(args.base_fname)