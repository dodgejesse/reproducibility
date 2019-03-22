import glob
import os
import argparse
from typing import Optional
from collections import ChainMap
import pandas as pd
import json


def get_argument_parser() -> Optional[argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--logdir",
        required=True,
    )
    
    return parser

if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    
    experiment_dir = os.path.abspath(args.logdir)
    metric_files = glob.glob(experiment_dir + '/run_*/trial/metrics.json')
    config_files = glob.glob(experiment_dir + '/run_*/trial/config.json')
    if not metric_files or not config_files:
        raise Exception("could not find files in experiment directory {}".format(experiment_dir))
    metrics = [json.load(open(f, 'r')) for f in metric_files]
    configs = [json.load(open(f, 'r')) for f in config_files]
    master = list(zip(configs, metrics))
    import ipdb; ipdb.set_trace()
    master_dicts = [dict(ChainMap(*item)) for item in master]

    df = pd.io.json.json_normalize(master_dicts)
    df.columns = df.columns.map(lambda x: x.split(".")[-1])
    output_file = os.path.join(experiment_dir, "results.tsv")
    df.to_csv(output_file, sep='\t')
    print("results written to {}".format(output_file))
