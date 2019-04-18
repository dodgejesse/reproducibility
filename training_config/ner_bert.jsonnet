// Configuration for the NER model with ELMo, modified slightly from
// the version included in "Deep Contextualized Word Representations",
// (https://arxiv.org/abs/1802.05365).  Compared to the version in this paper,
// this configuration replaces the original Senna word embeddings with
// 50d GloVe embeddings.
//
// There is a trained model available at https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.30.tar.gz
// with test set F1 of 92.51 compared to the single model reported
// result of 92.22 +/- 0.10.


{

  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "coding_scheme": "BIOUL",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      },
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased"
    }
    }
  },
  "train_data_path": "s3://suching-dev/ner-2003/train.txt",
  "validation_data_path": "s3://suching-dev/ner-2003/dev.txt",
  "test_data_path": "s3://suching-dev/ner-2003/test.txt",
  "evaluate_on_test": std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1,
  "model": {
    "type": "crf_tagger",
    "label_encoding": "BIOUL",
    "dropout": std.extVar("DROPOUT"),
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 50,
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
            "trainable": true
        },
        "bert": {
            "type": "bert-pretrained-enhanced",
            "pretrained_model": "bert-base-uncased",
            "requires_grad": false,
            "dropout": std.extVar("BERT_DROPOUT"),
            "first_layer_only": std.parseInt(std.extVar("FIRST_LAYER_ONLY")) == 1,
            "second_to_last_layer_only": std.parseInt(std.extVar("SECOND_TO_LAST_LAYER_ONLY")) == 1,
            "last_layer_only": std.parseInt(std.extVar("LAST_LAYER_ONLY")) == 1,
            "sum_last_four_layers": std.parseInt(std.extVar("SUM_LAST_FOUR_LAYERS")) == 1,
            "concat_last_four_layers": std.parseInt(std.extVar("CONCAT_LAST_FOUR_LAYERS")) == 1,
            "sum_all_layers": std.parseInt(std.extVar("SUM_ALL_LAYERS")) == 1,
            "scalar_mix": std.parseInt(std.extVar("SCALAR_MIX")) == 1,
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "embedding_dim": std.parseInt(std.extVar("CHARACTER_EMBEDDING_DIM"))
            },
            "encoder": {
            "type": "cnn",
            "embedding_dim": std.parseInt(std.extVar("CHARACTER_EMBEDDING_DIM")),
            "num_filters": std.parseInt(std.extVar("NUM_FILTERS")),
            "ngram_filter_sizes": std.range(1, std.parseInt(std.extVar("MAX_FILTER_SIZE"))),
            "conv_layer_activation": "relu"
            }
        }
      },
      "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"],
            "tokens": ["tokens"],
            "token_characters": ["token_characters"]
        }
    },
    "encoder": {
      "type": "lstm",
      "input_size": if std.parseInt(std.extVar("CONCAT_LAST_FOUR_LAYERS")) == 1 then 768 * 4 + 50 + std.parseInt(std.extVar("NUM_FILTERS")) * std.parseInt(std.extVar("MAX_FILTER_SIZE")) else 768  + 50 + std.parseInt(std.extVar("NUM_FILTERS")) * std.parseInt(std.extVar("MAX_FILTER_SIZE")),
      "hidden_size": std.parseInt(std.extVar("ENCODER_HIDDEN_SIZE")),
      "num_layers": std.parseInt(std.extVar("NUM_ENCODER_LAYERS")),
      "dropout": std.extVar("ENCODER_DROPOUT"),
      "bidirectional": true
    },
    "regularizer": [
      [
        "scalar_parameters",
        {
          "type": "l2",
          "alpha": std.extVar("ALPHA")
        }
      ]
    ]
  },
  "iterator": {
    "type": "basic",
    "batch_size": std.parseInt(std.extVar("BATCH_SIZE"))
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": std.extVar("LEARNING_RATE")
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 1,
    "num_epochs": 75,
    "grad_norm": std.extVar("GRAD_NORM"),
    "patience": 25,
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 2
    },
    "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE"))
  }
}