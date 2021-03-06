// GPU to use. Setting this to -1 will mean that we'll use the CPU.
local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));

// learning rate of overall model.
local LEARNING_RATE = std.extVar("LEARNING_RATE");

local BATCH_SIZE = std.parseInt(std.extVar("BATCH_SIZE"));

local EVALUATE_ON_TEST = std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1;

local NUM_EPOCHS = std.parseInt(std.extVar("NUM_EPOCHS"));

local GRAD_NORM = std.parseInt(std.extVar("GRAD_NORM"));

{
  "numpy_seed": std.extVar("SEED"),
  "pytorch_seed": std.extVar("SEED"),
  "random_seed": std.extVar("SEED"),
  // Slightly modified version of the bi-attentative classification model with ELMo
  // from "Deep contextualized word representations" (http://www.aclweb.org/anthology/N18-1202),
  // trained on 5-class Stanford Sentiment Treebank.
  // There is a trained model available at https://s3-us-west-2.amazonaws.com/allennlp/models/sst-5-elmo-biattentive-classification-network-2018.09.04.tar.gz
  // with test accuracy of 54.7%.
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": true,
    "granularity": "2-class",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
      "elmo": {
        "type": "elmo_characters"
      }
    }
  },
  "validation_dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "2-class",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
      # If using ELMo in the BCN, add an elmo_characters
      # token_indexers.
      "elmo": {
        "type": "elmo_characters"
      }
    }
  },

  "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt",
  "test_data_path": if EVALUATE_ON_TEST then "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt" else null,
  "evaluate_on_test": EVALUATE_ON_TEST,
  "model": {
    "type": "bcn",
    # The BCN model will consume the arrays generated by the ELMo token_indexer
    # independently of the text_field_embedder, so we do not include the elmo key here.
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
            "type": "embedding",
            "embedding_dim": 300,
            "trainable": true
        }
      }
    },
    "embedding_dropout": std.parseInt(std.extVar("EMBEDDING_DROPOUT")) / 10,
    "pre_encode_feedforward": {
        "input_dim": 1324,
        "num_layers": std.parseInt(std.extVar("PRE_ENCODE_FEEDFORWARD_LAYERS")),
        "hidden_dims": std.makeArray(std.parseInt(std.extVar("PRE_ENCODE_FEEDFORWARD_LAYERS")), function(i) std.parseInt(std.extVar("PRE_ENCODE_FEEDFORWARD_HIDDEN_DIMS"))),
        "activations": std.extVar("PRE_ENCODE_FEEDFORWARD_ACTIVATION"),
        "dropout": std.parseInt(std.extVar("PRE_ENCODE_FEEDFORWARD_DROPOUT")) / 10
    },
    "encoder": {
      "type": "lstm",
      "input_size": std.parseInt(std.extVar("PRE_ENCODE_FEEDFORWARD_HIDDEN_DIMS")),
      "hidden_size": std.parseInt(std.extVar("ENCODER_HIDDEN_SIZE")),
      "num_layers": std.parseInt(std.extVar("ENCODER_NUM_LAYERS")),
      "bidirectional": true
    },
    "integrator": {
      "type": "lstm",
      "input_size": std.parseInt(std.extVar("ENCODER_HIDDEN_SIZE")) * 2 * 3,
      "hidden_size": std.parseInt(std.extVar("INTEGRATOR_HIDDEN_SIZE")),
      "num_layers": std.parseInt(std.extVar("INTEGRATOR_NUM_LAYERS")),
      "bidirectional": true
    },
    "integrator_dropout": std.parseInt(std.extVar("INTEGRATOR_DROPOUT")) / 10,
    "elmo": {
      "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
      "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
      "do_layer_norm": false,
      "dropout": std.parseInt(std.extVar("ELMO_DROPOUT")) / 10,
      "num_output_representations": if std.parseInt(std.extVar("USE_INTEGRATOR_OUTPUT_ELMO")) == 0 then 1 else 2,
      "requires_grad": std.parseInt(std.extVar("FREEZE_ELMO")) == 0
    },
    "use_input_elmo": true,
    "use_integrator_output_elmo": std.parseInt(std.extVar("USE_INTEGRATOR_OUTPUT_ELMO")) == 1,
    "output_layer": {
        "input_dim": if std.parseInt(std.extVar("USE_INTEGRATOR_OUTPUT_ELMO")) == 1 then ( std.parseInt(std.extVar("INTEGRATOR_HIDDEN_SIZE")) * 2 + 1024 ) * 4 else std.parseInt(std.extVar("INTEGRATOR_HIDDEN_SIZE")) * 4 * 2,
        "num_layers": std.parseInt(std.extVar("OUTPUT_NUM_LAYERS")) + 1,
        "output_dims": std.makeArray(std.parseInt(std.extVar("OUTPUT_NUM_LAYERS")), function(i) std.parseInt(std.extVar("OUTPUT_DIM"))) + [2],
        "pool_sizes": std.parseInt(std.extVar("POOL_SIZES")),
        "dropout": std.makeArray(std.parseInt(std.extVar("OUTPUT_NUM_LAYERS")), function(i) std.parseInt(std.extVar("OUTPUT_DROPOUT")) / 10) + [0.0],
    },

  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : BATCH_SIZE
  },
  "trainer": {
    "num_epochs": NUM_EPOCHS,
    "patience": 10,
    "grad_norm": GRAD_NORM,
    "validation_metric": "+accuracy",
    "cuda_device": CUDA_DEVICE,
    "optimizer": {
      "type": "adam",
      "lr": LEARNING_RATE
    },
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 2
      },
      "num_serialized_models_to_keep": 1
  }
}