import argparse
import collections
from logging import warn
import os
import re
import warnings

import torch
from fairseq.file_io import PathManager


def average_checkpoints(inputs, key_filter):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.
      key_filter: A boolean function that filters parameter names

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for fpath in inputs:
        with PathManager.open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state["model"]
        if key_filter is not None:
            model_params_keys = [k for k in model_params if key_filter(k)]
        else:
            model_params_keys = list(model_params.keys())

        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
            else:
                params_dict[k] += p

    # averaged_params = collections.OrderedDict()
    with PathManager.open(inputs[0], "rb") as f:
        averaged_params = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )["model"]
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def last_n_checkpoints(paths, n, update_based, upper_bound=None):
    assert len(paths) == 1
    path = paths[0]
    if update_based:
        pt_regexp = re.compile(r"checkpoint_\d+_(\d+)\.pt")
    else:
        pt_regexp = re.compile(r"checkpoint(\d+)\.pt")
    files = PathManager.ls(path)

    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        if m is not None:
            sort_key = int(m.group(1))
            if upper_bound is None or sort_key <= upper_bound:
                entries.append((sort_key, m.group(0)))
    if len(entries) < n:
        # raise Exception(
        warnings.warn(
            "Found {} checkpoint files but need at least {}".format(len(entries), n)
        )
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)[:n]]


def main():
    parser = argparse.ArgumentParser(
        description="Tool to average the params of input checkpoints to "
                    "produce a new checkpoint",
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')
    parser.add_argument('--prefix', type=str, nargs='+',
                        help='List of key prefixes to filter parameters.')
    num_group = parser.add_mutually_exclusive_group()
    num_group.add_argument('--num-epoch-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_xx.pt in the path specified by input, '
                                'and average last this many of them.')
    num_group.add_argument('--num-update-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_ee_xx.pt in the path specified by input, '
                                'and average last this many of them.')
    parser.add_argument('--checkpoint-upper-bound', type=int,
                        help='when using --num-epoch-checkpoints, this will set an upper bound on which epoch to use, '
                             'when using --num-update-checkpoints, this will set an upper bound on which update to use'
                             'e.g., with --num-epoch-checkpoints=10 --checkpoint-upper-bound=50, checkpoints 41-50 would be averaged.'
                             'e.g., with --num-update-checkpoints=10 --checkpoint-upper-bound=50000, checkpoints 40500-50000 would be averaged assuming --save-interval-updates 500'
                        )
    args = parser.parse_args()
    print(args)

    num = None
    is_update_based = False
    if args.num_update_checkpoints is not None:
        num = args.num_update_checkpoints
        is_update_based = True
    elif args.num_epoch_checkpoints is not None:
        num = args.num_epoch_checkpoints

    assert args.checkpoint_upper_bound is None or (
            args.num_epoch_checkpoints is not None
            or args.num_update_checkpoints is not None
    ), "--checkpoint-upper-bound requires --num-epoch-checkpoints or --num-update-checkpoints"
    assert (
            args.num_epoch_checkpoints is None or args.num_update_checkpoints is None
    ), "Cannot combine --num-epoch-checkpoints and --num-update-checkpoints"

    if num is not None:
        args.inputs = last_n_checkpoints(
            args.inputs,
            num,
            is_update_based,
            upper_bound=args.checkpoint_upper_bound,
        )
        print("averaging checkpoints: ", args.inputs)

    key_filter = lambda x: any([x.startswith(p) for p in args.prefix]) if args.prefix else None
    new_state = average_checkpoints(args.inputs, key_filter)
    with PathManager.open(args.output, "wb") as f:
        torch.save(new_state, f)
    print("Finished writing averaged checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()
