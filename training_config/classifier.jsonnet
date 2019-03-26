// GPU to use. Setting this to -1 will mean that we'll use the CPU.
local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));

// Paths to data.
local TRAIN_PATH = std.extVar("DATA_DIR") + "train.jsonl";
local DEV_PATH =  std.extVar("DATA_DIR") + "dev.jsonl";

// Throttle the training data to a random subset of this length.
local THROTTLE = std.extVar("THROTTLE");

// Use the SpaCy tokenizer when reading in the data. If this is false, we'll use the just_spaces tokenizer.
local USE_SPACY_TOKENIZER = std.parseInt(std.extVar("USE_SPACY_TOKENIZER"));

// learning rate of overall model.
local LEARNING_RATE = std.extVar("LEARNING_RATE");

// dropout applied after pooling
local DROPOUT = std.parseInt(std.extVar("DROPOUT"));

local BATCH_SIZE = std.parseInt(std.extVar("BATCH_SIZE"));


local BOE_FIELDS(embedding_dim, averaged) = {
    "type": "seq2vec",
    "architecture": {
        "embedding_dim": embedding_dim,
        "type": "boe",
        "averaged": averaged
    }
};

local LSTM_FIELDS(num_encoder_layers, embedding_dim, hidden_size, aggregations) = {
      "type" : "seq2seq",
      "architecture": {
        "type": "lstm",
        "num_layers": num_encoder_layers,
        "bidirectional": true,
        "input_size": embedding_dim,
        "hidden_size": hidden_size
      },
      "aggregations": aggregations
};

local CNN_FIELDS(max_filter_size, embedding_dim, hidden_size, num_filters) = {
      "type": "seq2vec",
      "architecture": {
          "type": "cnn",
          "ngram_filter_sizes": std.range(1, max_filter_size),
          "num_filters": num_filters,
          "embedding_dim": embedding_dim,
          "output_dim": hidden_size, 
      },
};

local MAXPOOL_FIELDS(embedding_dim) = {
    "type": "seq2vec",
    "architecture": {
        "type": "maxpool",
        "embedding_dim": embedding_dim
    }
};

local CLS_TOKEN_FIELDS(embedding_dim) = {
    "type": "seq2vec",
    "architecture": {
        "type": "cls_token",
        "embedding_dim": embedding_dim
    }
};


local ELMO_LSTM_FIELDS = {
  "elmo_lstm_indexer": {
    "elmo": {
      "type": "elmo_characters",
    }
  },
  "elmo_lstm_embedder": {
    "elmo": {
      "type": "elmo_token_embedder",
      "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
      "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
      "do_layer_norm": false,
      "dropout": 0.0
    }
  },
  "embedding_dim": 1024
};

local BERT_FIELDS = {
  "bert_indexer": {
       "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased"
    }
  },
  "bert_embedder": {
    "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "requires_grad": true,
        "top_layer_only": false
        }
  },
  "extra_embedder_fields": {
    "allow_unmatched_keys": true,
    "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets"],
        "tokens": ["tokens"]
    }
  },
  "embedding_dim": 768
};

local ELMO_TRANSFORMER_FIELDS = {
  "elmo_transformer_indexer": {
    "elmo": {
      "type": "elmo_characters",
    }
  },
  "elmo_transformer_embedder": {
    "elmo": {
      "type": "bidirectional_lm_token_embedder",
      "archive_file": "s3://allennlp/models/transformer-elmo-2019.01.10.tar.gz",
      "dropout": 0.0,
      "bos_eos_tokens": ["<S>", "</S>"],
      "remove_bos_eos": true,
      "requires_grad": true
    }
  },
  "embedding_dim": 1024
};

local GLOVE_FIELDS = {
  "glove_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
    }
  },
  "glove_embedder": {
    "tokens": {
        "embedding_dim": 50,
        "trainable": true,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
    }
  },
  "embedding_dim": 50
};

local W2V_FIELDS = {
  "w2v_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": false,
    }
  },
  "w2v_embedder": {
    "tokens": {
        "embedding_dim": 300,
        "trainable": true,
        "pretrained_file": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
    }
  },
  "embedding_dim": 300
};

local CHAR_CNN_FIELDS = {
  "cnn_indexer": {
    "token_characters": {
      "type": "characters",
      "min_padding_length": "5",      
    }
  },
  "cnn_embedder": {
    "token_characters": {
      "type": "character_encoding",
        "embedding": {
          "trainable": true,
          "embedding_dim": 32
      },
      "encoder": {
        "type": "cnn",
        "embedding_dim": 32,
        "num_filters": 100,
        "ngram_filter_sizes": [2,3,4,5]
      },
    }
  },
  "embedding_dim": 400
};

local CHAR_LSTM_FIELDS = {
  "lstm_indexer": {
    "token_characters": {
      "type": "characters",
      "min_padding_length": "5",      
    }
  },
  "lstm_embedder": {
    "token_characters": {
      "type": "character_encoding",
        "embedding": {
          "trainable": true,
          "embedding_dim": 32
      },
      "encoder": {
        "type": "lstm",
        "input_size": 32,
        "num_layers": 1,
        "bidirectional": true,
        "hidden_size": 200
      },
    }
  },
  "embedding_dim": 400
};

local RANDOM_FIELDS = {
  "random_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
    }
  },
  "random_embedder": {
    "tokens": {
        "embedding_dim": 300,
        "trainable": true,
        "type": "embedding",
    }
  },
  "embedding_dim": 300
};


local BOW_COUNT_FIELDS = {
  "bow_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
    }
  },
  "bow_embedder": {
    "tokens": {
        "type": "bag_of_word_counts",
    }
  },
  "embedding_dim": 0
};



local TOKEN_INDEXERS = if std.extVar("EMBEDDING") == "ELMO_LSTM" then ELMO_LSTM_FIELDS['elmo_lstm_indexer'] else {}
                        + if std.extVar("EMBEDDING") == "BERT" then BERT_FIELDS['bert_indexer'] else {}
                        + if std.extVar("EMBEDDING") == "RANDOM" then RANDOM_FIELDS['random_indexer'] else {}
                        + if std.extVar("EMBEDDING") == "W2V" then W2V_FIELDS['w2v_indexer'] else {}
                        + if std.extVar("EMBEDDING") == "CHAR_CNN" then CHAR_CNN_FIELDS['cnn_indexer'] else {}
                        + if std.extVar("EMBEDDING") == "CHAR_LSTM" then CHAR_LSTM_FIELDS['lstm_indexer'] else {}
                        + if std.extVar("EMBEDDING") == "GLOVE" then GLOVE_FIELDS['glove_indexer'] else {}
                        + if std.extVar("EMBEDDING") == "ELMO_TRANSFORMER" then ELMO_TRANSFORMER_FIELDS['elmo_transformer_indexer'] else {}
                        + if std.extVar("EMBEDDING") == "BOW_COUNTS" then BOW_COUNT_FIELDS['bow_indexer'] else {};


local TOKEN_EMBEDDERS = if std.extVar("EMBEDDING") == "ELMO_LSTM" then ELMO_LSTM_FIELDS['elmo_lstm_embedder'] else {}
                        + if std.extVar("EMBEDDING") == "BERT" then BERT_FIELDS['bert_embedder'] else {}
                        + if std.extVar("EMBEDDING") == "RANDOM" then RANDOM_FIELDS['random_embedder'] else {}
                        + if std.extVar("EMBEDDING") == "W2V" then W2V_FIELDS['w2v_embedder'] else {}
                        + if std.extVar("EMBEDDING") == "CHAR_CNN" then CHAR_CNN_FIELDS['cnn_embedder'] else {}
                        + if std.extVar("EMBEDDING") == "CHAR_LSTM" then CHAR_LSTM_FIELDS['lstm_embedder'] else {}
                        + if std.extVar("EMBEDDING") == "GLOVE" then GLOVE_FIELDS['glove_embedder'] else {}
                        + if std.extVar("EMBEDDING") == "ELMO_TRANSFORMER" then ELMO_TRANSFORMER_FIELDS['elmo_transformer_embedder'] else {}
                        + if std.extVar("EMBEDDING") == "BOW_COUNTS" then BOW_COUNT_FIELDS['bow_embedder'] else {};

local EMBEDDING_DIM = if std.extVar("EMBEDDING") == "ELMO_LSTM" then ELMO_LSTM_FIELDS['embedding_dim'] else 0
                        + if std.extVar("EMBEDDING") == "BERT" then BERT_FIELDS['embedding_dim'] else 0
                        + if std.extVar("EMBEDDING") == "RANDOM" then RANDOM_FIELDS['embedding_dim'] else 0
                        + if std.extVar("EMBEDDING") == "W2V" then W2V_FIELDS['embedding_dim'] else 0
                        + if std.extVar("EMBEDDING") == "CHAR_CNN" then CHAR_CNN_FIELDS['embedding_dim'] else 0
                        + if std.extVar("EMBEDDING") == "CHAR_LSTM" then CHAR_LSTM_FIELDS['embedding_dim'] else 0
                        + if std.extVar("EMBEDDING") == "GLOVE" then GLOVE_FIELDS['embedding_dim'] else 0
                        + if std.extVar("EMBEDDING") == "ELMO_TRANSFORMER" then ELMO_TRANSFORMER_FIELDS['embedding_dim'] else 0;



local ENCODER = if std.extVar("ENCODER") == "AVERAGE" then BOE_FIELDS(EMBEDDING_DIM, true) else {} + 
                if std.extVar("ENCODER") == "SUM" then BOE_FIELDS(EMBEDDING_DIM, false) else {} + 
                if std.extVar("ENCODER") == "MAXPOOL" then MAXPOOL_FIELDS(EMBEDDING_DIM) else {} + 
                if std.extVar("ENCODER") == "CLS_TOKEN" then CLS_TOKEN_FIELDS(EMBEDDING_DIM) else {} + 
                if std.extVar("ENCODER") == "LSTM" then LSTM_FIELDS(std.parseInt(std.extVar("NUM_ENCODER_LAYERS")), EMBEDDING_DIM, std.parseInt(std.extVar("HIDDEN_SIZE")), std.extVar("AGGREGATIONS")) else {} +
                if std.extVar("ENCODER") == "CNN" then CNN_FIELDS(std.parseInt(std.extVar("MAX_FILTER_SIZE")), EMBEDDING_DIM, std.parseInt(std.extVar("HIDDEN_SIZE")), std.extVar("NUM_FILTERS")) else {};



local BASE_READER(TOKEN_INDEXERS, THROTTLE, USE_SPACY_TOKENIZER) = {
  "lazy": false,
  "type": "semisupervised_text_classification_json",
  "tokenizer": {
    "word_splitter": if USE_SPACY_TOKENIZER == 1 then "spacy" else "just_spaces",
  },
  "token_indexers": TOKEN_INDEXERS,
  "sample": THROTTLE,
};

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": BASE_READER(TOKEN_INDEXERS, THROTTLE, USE_SPACY_TOKENIZER),
   "validation_dataset_reader": BASE_READER(TOKEN_INDEXERS, null, USE_SPACY_TOKENIZER),
  // NOTE: we are assuming that vocabulary is created from both train, dev, and test. 
  // Our data splitting should ensure that there is no overlap between these splits.
  //  "datasets_for_vocab_creation": ["train"],
   "train_data_path": TRAIN_PATH,
   "validation_data_path": DEV_PATH,
   "model": {
      "type": "classifier",
      "input_embedder": {
                "token_embedders": TOKEN_EMBEDDERS
      } + if std.extVar("EMBEDDING") == "BERT" then BERT_FIELDS['extra_embedder_fields'] else {},
      "encoder": if std.extVar("EMBEDDING") == "BOW_COUNTS" then null else ENCODER,
      "dropout": DROPOUT
   },	
    "iterator": {
      "batch_size": BATCH_SIZE,
      "type": "basic"
   },
   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_epochs": 50,
      "optimizer": {
         "lr": LEARNING_RATE,
         "type": "adam"
      },
      "patience": 5,
      "validation_metric": "+accuracy",
      "num_serialized_models_to_keep": 1
   }
}
