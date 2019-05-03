{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
    "dataset_reader": {
        "type": "entailment_tuple",
        "max_tokens": 500,
        "max_tuples": 500,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        },
        "tokenizer": {
            "end_tokens": []
        }
    },
  "train_data_path": "/home/suching/reproducibility/scitail/SciTailV1.1/dgem_format/scitail_1.0_structure_train.tsv",
  "validation_data_path": "/home/suching/reproducibility/scitail/SciTailV1.1/dgem_format/scitail_1.0_structure_dev.tsv",
  "test_data_path": "/home/suching/reproducibility/scitail/SciTailV1.1/dgem_format/scitail_1.0_structure_test.tsv",
    "model": {
        "type": "tree_attention",
        "use_encoding_for_node": false,
        "ignore_edges": false,
        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "projection_dim": std.parseInt(std.extVar("PROJECTION_DIM")),
                "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
                "embedding_dim": 300,
                "trainable": false
            }
        },
        "attention_similarity": {
            "type": "dot_product"
        },
        "premise_encoder": {
            "type": "lstm",
            "bidirectional": std.parseInt(std.extVar("PREMISE_ENCODER_BIDIRECTIONAL")) == 1,
            "num_layers": std.parseInt(std.extVar("PREMISE_ENCODER_NUM_LAYERS")),
            "input_size": std.parseInt(std.extVar("PROJECTION_DIM")),
            "hidden_size": std.parseInt(std.extVar("PREMISE_ENCODER_HIDDEN_SIZE"))
        },
        "phrase_probability": {
            "input_dim": std.parseInt(std.extVar("PROJECTION_DIM")) * 4,
            "num_layers": std.parseInt(std.extVar("PHRASE_PROBABILITY_NUM_LAYERS"))  + 1,
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("PHRASE_PROBABILITY_NUM_LAYERS")), function(i) std.parseInt(std.extVar("PHRASE_PROBABILITY_HIDDEN_DIMS"))) + [2],
            "activations": std.makeArray(std.parseInt(std.extVar("PHRASE_PROBABILITY_NUM_LAYERS")), function(i) std.extVar("PHRASE_PROBABILITY_ACTIVATION")) + ['linear'],
            "dropout": std.makeArray(std.parseInt(std.extVar("PHRASE_PROBABILITY_NUM_LAYERS")), function(i) std.extVar("PHRASE_PROBABILITY_DROPOUT")) + ['0'],
        },
        "edge_probability": {
            "input_dim": if std.parseInt(std.extVar("PREMISE_ENCODER_BIDIRECTIONAL")) == 1 then std.parseInt(std.extVar("PREMISE_ENCODER_HIDDEN_SIZE")) * 4 + std.parseInt(std.extVar("EDGE_EMBEDDING_SIZE")) else std.parseInt(std.extVar("PREMISE_ENCODER_HIDDEN_SIZE")) * 2 + std.parseInt(std.extVar("EDGE_EMBEDDING_SIZE")),
            "num_layers": std.parseInt(std.extVar("EDGE_PROBABILITY_NUM_LAYERS"))  + 1,
            "hidden_dims": std.makeArray(std.parseInt(std.extVar("EDGE_PROBABILITY_NUM_LAYERS")), function(i) std.parseInt(std.extVar("EDGE_PROBABILITY_HIDDEN_DIMS"))) + [2],
            "activations": std.makeArray(std.parseInt(std.extVar("EDGE_PROBABILITY_NUM_LAYERS")), function(i) std.extVar("EDGE_PROBABILITY_ACTIVATION")) + ['linear'],
            "dropout": std.makeArray(std.parseInt(std.extVar("EDGE_PROBABILITY_NUM_LAYERS")), function(i) std.extVar("EDGE_PROBABILITY_DROPOUT")) + ['0.0'],
        },
        "edge_embedding": {
            "vocab_namespace": "edges",
            "embedding_dim": std.parseInt(std.extVar("EDGE_EMBEDDING_SIZE"))
        },
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_normal"}],
            [".*token_embedder_tokens._projection.*weight", {"type": "xavier_normal"}]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
        "batch_size": std.parseInt(std.extVar("BATCH_SIZE"))
    },

    "trainer": {
        "num_epochs": std.parseInt(std.extVar("NUM_EPOCHS")),
        "grad_norm": std.extVar("GRAD_NORM"),
        "patience": 20,
        "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "adam",
            "lr": std.extVar("LEARNING_RATE")
        },
        "learning_rate_scheduler": {
            "type": "exponential",
            "gamma": 0.5
        }
    }
}
