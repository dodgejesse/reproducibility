// Configuraiton for a textual entailment model based on:
//  Parikh, Ankur P. et al. “A Decomposable Attention Model for Natural Language Inference.” EMNLP (2016).
{
  "dataset_reader": {
    "type": "snli",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "tokenizer": {
      "end_tokens": ["@@NULL@@"]
    }
  },
  "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_train.jsonl",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl",
  "test_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_test.jsonl",
  "evaluate_on_test": std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1,
  "model": {
    "type": "decomposable_attention",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "projection_dim": std.parseInt(std.extVar("PROJECTION_DIM")),
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
            "embedding_dim": 300,
            "trainable": false
        }
      }
    },
    "attend_feedforward": {
      "input_dim": std.parseInt(std.extVar("PROJECTION_DIM")),
      "num_layers": std.parseInt(std.extVar("ATTEND_FEEDFORWARD_NUM_LAYERS")),
      "hidden_dims": std.parseInt(std.extVar("ATTEND_FEEDFORWARD_HIDDEN_DIMS")),
      "activations": std.extVar("ATTEND_FEEDFORWARD_ACTIVATION"),
      "dropout": std.extVar("ATTEND_FEEDFORWARD_DROPOUT")
    },
    "similarity_function": {"type": "dot_product"},
    "compare_feedforward": {
      "input_dim": std.parseInt(std.extVar("PROJECTION_DIM")) * 2,
      "num_layers": std.parseInt(std.extVar("COMPARE_FEEDFORWARD_NUM_LAYERS")),
      "hidden_dims": std.parseInt(std.extVar("COMPARE_FEEDFORWARD_HIDDEN_DIMS")),
      "activations": std.extVar("COMPARE_FEEDFORWARD_ACTIVATION"),
      "dropout": std.extVar("COMPARE_FEEDFORWARD_DROPOUT")
    },
    "aggregate_feedforward": {
      "input_dim": std.parseInt(std.extVar("COMPARE_FEEDFORWARD_HIDDEN_DIMS")) * 2,
      "num_layers": std.parseInt(std.extVar("AGGREGATE_FEEDFORWARD_NUM_LAYERS")) + 1,
      "hidden_dims": std.makeArray(std.parseInt(std.extVar("AGGREGATE_FEEDFORWARD_NUM_LAYERS")), function(i) std.parseInt(std.extVar("AGGREGATE_FEEDFORWARD_HIDDEN_DIMS"))) + [3],
      "activations": std.makeArray(std.parseInt(std.extVar("AGGREGATE_FEEDFORWARD_NUM_LAYERS")), function(i) std.extVar("AGGREGATE_FEEDFORWARD_ACTIVATION")) + ["linear"],
      "dropout": std.makeArray(std.parseInt(std.extVar("AGGREGATE_FEEDFORWARD_NUM_LAYERS")), function(i) std.extVar("AGGREGATE_FEEDFORWARD_DROPOUT")) + ["0.0"]
    },
     "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*token_embedder_tokens\\._projection.*weight", {"type": "xavier_normal"}]
     ]
   },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
    "batch_size": std.parseInt(std.extVar("BATCH_SIZE"))
  },

  "trainer": {
    "num_epochs": std.parseInt(std.extVar("NUM_EPOCHS")),
    "patience": 20,
    "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
    "grad_clipping": std.extVar("GRAD_CLIP"),
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad",
      "lr": std.extVar("LEARNING_RATE")
    }
  }
}