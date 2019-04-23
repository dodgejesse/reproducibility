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
      "elmo": {
        "type": "elmo_characters"
     }
    }
  },
  "train_data_path": "s3://suching-dev/ner-2003/train.txt",
  "validation_data_path": "s3://suching-dev/ner-2003/dev.txt",
  "test_data_path": "s3://suching-dev/ner-2003/test.txt",
  "evaluate_on_test": std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1,
  "model": {
    "type": "crf_tagger_elmo",
    "label_encoding": "BIOUL",
    "dropout": std.extVar("DROPOUT"),
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "elmo":{
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": false,
            "dropout": std.extVar("ELMO_DROPOUT"),
            "requires_grad": true
        }
      }
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