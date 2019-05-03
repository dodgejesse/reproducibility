import argparse
import json

from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from tqdm import tqdm
from nltk.util import skipgrams, ngrams
import numpy as np

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=str,
                        help="Path to the IMDB jsonl file.")
    parser.add_argument("--save-path", type=str,
                        help="Path to store the preprocessed corpus (output file name).")
    args = parser.parse_args()

    tokenizer = SpacyWordSplitter()
    tokenized_examples = []
    with tqdm(open(args.data_path, "r")) as f:
        for line in f:
            example = json.loads(line)
            tokens = list(map(str, tokenizer.split_words(example['text'])))
            example['text'] = ' '.join(tokens)
            # premise_tokens = list(map(str, tokenizer.split_words(example['sentence1'])))
            # hypothesis_tokens = list(map(str, tokenizer.split_words(example['sentence2'])))
            # example['sentence1'] = ' '.join(premise_tokens)
            # example['sentence2'] = ' '.join(hypothesis_tokens)
            # prem_trigrams = set(skipgrams(premise_tokens, 3, 1))
            # prem_bigrams = set(skipgrams(premise_tokens, 2, 1))
            # prem_unigrams = set(ngrams(premise_tokens, 1))

            # # n-grams from the hypothesis
            # hyp_trigrams = set(skipgrams(hypothesis_tokens, 3, 1))
            # hyp_bigrams = set(skipgrams(hypothesis_tokens, 2, 1))
            # hyp_unigrams = set(ngrams(hypothesis_tokens, 1))

            # # overlap proportions
            # if hyp_trigrams:
            #     tri_overlap = len(prem_trigrams.intersection(hyp_trigrams)) / len(hyp_trigrams)
            # else:
            #     tri_overlap = 0.0
            # if hyp_bigrams:
            #     bi_overlap = len(prem_bigrams.intersection(hyp_bigrams)) / len(hyp_bigrams)
            # else:
            #     bi_overlap = 0.0
            # if hyp_unigrams:
            #     uni_overlap = len(prem_unigrams.intersection(hyp_unigrams)) / len(hyp_unigrams)
            # else:
            #     uni_overlap = 0.0

            # example['features'] = [tri_overlap, bi_overlap, uni_overlap]
            tokenized_examples.append(example)
    write_jsons_to_file(tokenized_examples, args.save_path)


def write_jsons_to_file(jsons, save_path):
    """
    Write each json object in 'jsons' as its own line in the file designated by 'save_path'.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    out_file = open(save_path, "a")
    for example in tqdm(jsons):
        json.dump(example, out_file, ensure_ascii=False)
        out_file.write('\n')

if __name__ == '__main__':
    main()
