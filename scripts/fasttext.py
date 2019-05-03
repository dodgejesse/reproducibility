import fasttext
from environments import ENVIRONMENTS
from environments.random_search import HyperparameterSearch
import os
import shutil
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-o',
                        '--override',
                        action="store_true",
                        help='remove the specified serialization dir before training')
    parser.add_argument('-s', '--serialization_dir', type=str, help='model serialization directory', required=True)
    parser.add_argument('-e', '--environment', type=str, help='hyperparameter environment', required=True)
    parser.add_argument('-i', '--input', type=str, help='input file', required=False)
    parser.add_argument('-d', '--dev_file', type=str, help='dev file', required=False)
    parser.add_argument('-n', '--num_assignments', type=int, help='number of assignments', required=True)

    args = parser.parse_args()

    if os.path.exists(args.serialization_dir) and args.override:
        print(f"overriding {args.serialization_dir}")
        shutil.rmtree(args.serialization_dir)
    else:
        pathlib.Path(args.serialization_dir).mkdir(parents=True, exist_ok=True)

    dev = pd.read_csv(args.dev_file, sep='\t', names=['text', 'label'])
    
    df = pd.DataFrame()
    pbar = tqdm(range(args.num_assignments))
    for i in pbar:
        pathlib.Path(args.serialization_dir + f"/run_{i}/").mkdir(parents=True, exist_ok=True)
        env = ENVIRONMENTS[args.environment.upper()]
        space = HyperparameterSearch(**env)
        sample = space.sample()
        start = time.time()
        model_file = args.serialization_dir + f"/run_{i}/model"
        classifier = fasttext.supervised(args.input, model_file, **sample)
        end = time.time()
        result = classifier.test(args.dev_file)
        predicted_labels = classifier.predict(dev.text)
        for k, v in sample.items():
            sample[k] = [v]
        sub_df = pd.DataFrame(sample)
        sub_df['accuracy'] = (pd.Series(['__label__' + x for y in predicted_labels for x in y]) == dev.label).sum() / dev.shape[0] 
        sub_df['training_duration'] = end - start
        sub_df['precision'] = result.precision
        sub_df['recall'] = result.recall
        sub_df['f1'] = 2 * (result.precision * result.recall) / (result.precision + result.recall)
        df = pd.concat([df, sub_df], 0)
        best_trial = df.iloc[df.accuracy.idxmax()]
        pbar.set_description(f"best accuracy: {best_trial['accuracy']}, best F1: {best_trial['f1']}")
    df.to_json(args.serialization_dir + "/results.jsonl", lines=True, orient='records')
    print(f"best accuracy: {df.accuracy.max()}")