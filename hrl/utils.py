import os
import csv
import sys
import logging
from pydoc import locate
from collections import defaultdict
from distutils.util import strtobool


def create_log_dir(experiment_name):
    """
    Prepare a directory for outputting training results.
    Then the following infomation is saved into the directory:
        command.txt: command itself
    Additionally, if the current directory is under git control, the following
    information is saved:
        git-head.txt: result of `git rev-parse HEAD`
        git-status.txt: result of `git status`
        git-log.txt: result of `git log`
        git-diff.txt: result of `git diff HEAD`
    """
    outdir = os.path.join(os.getcwd(), experiment_name)
    # create log dir
    try:
        os.makedirs(outdir, exist_ok=False)
    except OSError:
        print(f"Creation of the directory {outdir} failed")
    else:
        print(f"Successfully created the directory {outdir}")

    # log the command used
    with open(os.path.join(outdir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv))

    # log git stuff
    from pfrl.experiments.prepare_output_dir import is_under_git_control, save_git_information
    if is_under_git_control():
        save_git_information(outdir)
    return outdir


def load_hyperparams(filepath):
    params = dict()
    with open(filepath, newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for name, value, dtype in reader:
            if dtype == 'bool':
                params[name] = bool(strtobool(value))
            else:
                params[name] = locate(dtype)(value)
    return params


def save_hyperparams(filepath, params):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for name, value in sorted(params.items()):
            type_str = defaultdict(lambda: None, {
                bool: 'bool',
                int: 'int',
                str: 'str',
                float: 'float',
            })[type(value)] # yapf: disable
            if type_str is not None:
                writer.writerow((name, value, type_str))


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def update_param(params, name, value):
    if name not in params:
        raise KeyError(
            "Parameter '{}' specified, but not found in hyperparams file.".format(name))
    else:
        logging.info("Updating parameter '{}' to {}".format(name, value))
    if type(params[name]) == bool:
        params[name] = bool(strtobool(value))
    else:
        params[name] = type(params[name])(value)
