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
    dirs = glob.glob(experiment_dir + '/run_*/trial/')

    master = []
    for dir in dirs:
        try:
            metric = json.load(open(os.path.join(dir, "metrics.json"), 'r'))
            config = json.load(open(os.path.join(dir, "config.json"), 'r'))
            master.append((metric, config))
        except:
            continue


    master_dicts = [dict(ChainMap(*item)) for item in master]

    df = pd.io.json.json_normalize(master_dicts)
    df['model.encoder.architecture.type'] = df['model.encoder.architecture.type'].fillna("linear")
    output_file = os.path.join(experiment_dir, "results.tsv")
    df.to_csv(output_file, sep='\t')
    print("results written to {}".format(output_file))
    print(f"total experiments: {df.shape[0]}")
    print(f"best models:\n{df.groupby('model.encoder.architecture.type').best_validation_accuracy.max()}")