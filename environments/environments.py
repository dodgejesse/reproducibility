
from environments.random_search import RandomSearch

DATA_DIR = "s3://suching-dev/final-datasets/sst/"

BOW_LINEAR = {
        "LAZY_DATASET_READER": 0,
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 0,
        "NUM_EPOCHS": 50,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100000),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "EMBEDDINGS": ["BOW_COUNTS"],
        "ENCODER": None,
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "DROPOUT": RandomSearch.random_integer(0, 5),
        "BATCH_SIZE": 64,
}


SST_LSTM_GLOVE_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 0,
        "NUM_EPOCHS": 50,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 0,
        "FREEZE_EMBEDDINGS": None,
        "EMBEDDINGS": ["GLOVE"],
        "ENCODER": "LSTM",
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "DROPOUT": RandomSearch.random_integer(0, 5),
        "BATCH_SIZE": 64,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "NUM_OUTPUT_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "MAX_FILTER_SIZE": RandomSearch.random_integer(3, 6),
        "NUM_FILTERS": RandomSearch.random_integer(64, 512),
        "HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "AGGREGATIONS": RandomSearch.random_subset("maxpool", "meanpool", "attention", "final_state"),
        "MAX_CHARACTER_FILTER_SIZE": RandomSearch.random_choice(3, 4, 5, 6),
        "NUM_CHARACTER_FILTERS": RandomSearch.random_integer(16, 64),
        "CHARACTER_HIDDEN_SIZE": RandomSearch.random_integer(16, 128),
        "CHARACTER_EMBEDDING_DIM": RandomSearch.random_integer(16, 64),
        "CHARACTER_ENCODER": RandomSearch.random_choice("LSTM", "CNN", "AVERAGE"),
        "NUM_CHARACTER_ENCODER_LAYERS": RandomSearch.random_choice(1, 2),
}

SST_FROZEN_ELMO_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 0,
        "NUM_EPOCHS": 50,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 0,
        "FREEZE_EMBEDDINGS": "ELMO_LSTM",
        "EMBEDDINGS": ["GLOVE", "ELMO_LSTM"],
        "ENCODER": "MAXPOOL",
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "DROPOUT": RandomSearch.random_integer(0, 5),
        "BATCH_SIZE": 64,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "NUM_OUTPUT_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "MAX_FILTER_SIZE": RandomSearch.random_integer(3, 6),
        "NUM_FILTERS": RandomSearch.random_integer(64, 512),
        "HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "AGGREGATIONS": RandomSearch.random_subset("maxpool", "meanpool", "attention", "final_state"),
        "MAX_CHARACTER_FILTER_SIZE": RandomSearch.random_choice(3, 4, 5, 6),
        "NUM_CHARACTER_FILTERS": RandomSearch.random_integer(16, 64),
        "CHARACTER_HIDDEN_SIZE": RandomSearch.random_integer(16, 128),
        "CHARACTER_EMBEDDING_DIM": RandomSearch.random_integer(16, 64),
        "CHARACTER_ENCODER": RandomSearch.random_choice("LSTM", "CNN", "AVERAGE"),
        "NUM_CHARACTER_ENCODER_LAYERS": RandomSearch.random_choice(1, 2),
}

SST_FINE_TUNED_ELMO_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 0,
        "NUM_EPOCHS": 50,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 0,
        "FREEZE_EMBEDDINGS": None,
        "EMBEDDINGS": ["ELMO_LSTM"],
        "ENCODER": "MAXPOOL",
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "DROPOUT": RandomSearch.random_integer(0, 5),
        "BATCH_SIZE": 64,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "NUM_OUTPUT_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "MAX_FILTER_SIZE": RandomSearch.random_integer(3, 6),
        "NUM_FILTERS": RandomSearch.random_integer(64, 512),
        "HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "AGGREGATIONS": RandomSearch.random_subset("maxpool", "meanpool", "attention", "final_state"),
        "MAX_CHARACTER_FILTER_SIZE": RandomSearch.random_choice(3, 4, 5, 6),
        "NUM_CHARACTER_FILTERS": RandomSearch.random_integer(16, 64),
        "CHARACTER_HIDDEN_SIZE": RandomSearch.random_integer(16, 128),
        "CHARACTER_EMBEDDING_DIM": RandomSearch.random_integer(16, 64),
        "CHARACTER_ENCODER": RandomSearch.random_choice("LSTM", "CNN", "AVERAGE"),
        "NUM_CHARACTER_ENCODER_LAYERS": RandomSearch.random_choice(1, 2),
}

CLASSIFIER_LSTM_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 0,
        "NUM_EPOCHS": 50,
        "SEED": RandomSearch.random_integer(0, 100000),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 0,
        "FREEZE_EMBEDDINGS": None,
        "EMBEDDINGS": ["GLOVE"],
        "ENCODER": "LSTM",
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "DROPOUT": RandomSearch.random_integer(0, 5),
        "BATCH_SIZE": 64,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "NUM_OUTPUT_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "MAX_FILTER_SIZE": RandomSearch.random_integer(3, 6),
        "NUM_FILTERS": RandomSearch.random_integer(64, 512),
        "HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "AGGREGATIONS": RandomSearch.random_subset("maxpool", "meanpool", "attention", "final_state"),
        "MAX_CHARACTER_FILTER_SIZE": RandomSearch.random_choice(3, 4, 5, 6),
        "NUM_CHARACTER_FILTERS": RandomSearch.random_integer(16, 64),
        "CHARACTER_HIDDEN_SIZE": RandomSearch.random_integer(16, 128),
        "CHARACTER_EMBEDDING_DIM": RandomSearch.random_integer(16, 64),
        "CHARACTER_ENCODER": RandomSearch.random_choice("LSTM", "CNN", "AVERAGE"),
        "NUM_CHARACTER_ENCODER_LAYERS": RandomSearch.random_choice(1, 2),
}

CLASSIFIER_CNN_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 0,
        "NUM_EPOCHS": 50,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 0,
        "FREEZE_EMBEDDINGS": None,
        "EMBEDDINGS": ["GLOVE"],
        "ENCODER": "CNN",
        "LEARNING_RATE": RandomSearch.random_uniform(1e-6, 1e-1),
        "DROPOUT": RandomSearch.random_integer(0, 5),
        "BATCH_SIZE": 64,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "NUM_OUTPUT_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "MAX_FILTER_SIZE": RandomSearch.random_integer(3, 6),
        "NUM_FILTERS": RandomSearch.random_integer(64, 512),
        "HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "AGGREGATIONS": RandomSearch.random_subset("maxpool", "meanpool", "attention", "final_state"),
        "MAX_CHARACTER_FILTER_SIZE": RandomSearch.random_choice(3, 4, 5, 6),
        "NUM_CHARACTER_FILTERS": RandomSearch.random_integer(16, 64),
        "CHARACTER_HIDDEN_SIZE": RandomSearch.random_integer(16, 128),
        "CHARACTER_EMBEDDING_DIM": RandomSearch.random_integer(16, 64),
        "CHARACTER_ENCODER": RandomSearch.random_choice("LSTM", "CNN", "AVERAGE"),
        "NUM_CHARACTER_ENCODER_LAYERS": RandomSearch.random_choice(1, 2),
}

CLASSIFIER_BOE_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 0,
        "NUM_EPOCHS": 50,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 0,
        "FREEZE_EMBEDDINGS": None,
        "EMBEDDINGS": ["GLOVE"],
        "ENCODER": "MAXPOOL",
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "DROPOUT": RandomSearch.random_integer(0, 5),
        "BATCH_SIZE": 64,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "NUM_OUTPUT_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "MAX_FILTER_SIZE": RandomSearch.random_integer(3, 6),
        "NUM_FILTERS": RandomSearch.random_integer(64, 512),
        "HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "AGGREGATIONS": RandomSearch.random_subset("maxpool", "meanpool", "attention", "final_state"),
        "MAX_CHARACTER_FILTER_SIZE": RandomSearch.random_choice(3, 4, 5, 6),
        "NUM_CHARACTER_FILTERS": RandomSearch.random_integer(16, 64),
        "CHARACTER_HIDDEN_SIZE": RandomSearch.random_integer(16, 128),
        "CHARACTER_EMBEDDING_DIM": RandomSearch.random_integer(16, 64),
        "CHARACTER_ENCODER": RandomSearch.random_choice("LSTM", "CNN", "AVERAGE"),
        "NUM_CHARACTER_ENCODER_LAYERS": RandomSearch.random_choice(1, 2),
}


DAM_SEARCH = {
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 1,
        "PROJECTION_DIM": RandomSearch.random_integer(64, 300),
        "ATTEND_FEEDFORWARD_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "ATTEND_FEEDFORWARD_HIDDEN_DIMS": RandomSearch.random_integer(64, 512),
        "ATTEND_FEEDFORWARD_ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
        "ATTEND_FEEDFORWARD_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "COMPARE_FEEDFORWARD_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "COMPARE_FEEDFORWARD_HIDDEN_DIMS": RandomSearch.random_integer(64, 512),
        "COMPARE_FEEDFORWARD_ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
        "COMPARE_FEEDFORWARD_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "AGGREGATE_FEEDFORWARD_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "AGGREGATE_FEEDFORWARD_HIDDEN_DIMS": RandomSearch.random_integer(64, 512),
        "AGGREGATE_FEEDFORWARD_ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
        "AGGREGATE_FEEDFORWARD_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "SEED": RandomSearch.random_integer(0, 100000),
        "GRAD_CLIP": RandomSearch.random_uniform(5, 10),
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "BATCH_SIZE": 64,
        "NUM_EPOCHS": 140
}

# DAM_SEARCH = {
#         "CUDA_DEVICE": 0,
#         "EVALUATE_ON_TEST": 1,
#         "PROJECTION_DIM": RandomSearch.random_integer(64, 300),
#         "ATTEND_FEEDFORWARD_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
#         "ATTEND_FEEDFORWARD_HIDDEN_DIMS": RandomSearch.random_integer(64, 512),
#         "ATTEND_FEEDFORWARD_ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
#         "ATTEND_FEEDFORWARD_DROPOUT": RandomSearch.random_uniform(0, 0.5),
#         "COMPARE_FEEDFORWARD_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
#         "COMPARE_FEEDFORWARD_HIDDEN_DIMS": RandomSearch.random_integer(64, 512),
#         "COMPARE_FEEDFORWARD_ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
#         "COMPARE_FEEDFORWARD_DROPOUT": RandomSearch.random_uniform(0, 0.5),
#         "AGGREGATE_FEEDFORWARD_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
#         "AGGREGATE_FEEDFORWARD_HIDDEN_DIMS": RandomSearch.random_integer(64, 512),
#         "AGGREGATE_FEEDFORWARD_ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
#         "AGGREGATE_FEEDFORWARD_DROPOUT": RandomSearch.random_uniform(0, 0.5),
#         "SEED": RandomSearch.random_integer(0, 100),
#         "GRAD_CLIP": RandomSearch.random_uniform(5, 10),
#         "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
#         "BATCH_SIZE": 64,
#         "NUM_EPOCHS": 140,
#         "DROPOUT": RandomSearch.random_uniform(0, 0.5)
# }

ESIM_SEARCH = {
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 1,
        "BATCH_SIZE": 32,
        "DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "ENCODER_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "ENCODER_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "PROJECTION_FEEDFORWARD_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "PROJECTION_FEEDFORWARD_HIDDEN_DIM": RandomSearch.random_integer(64, 512),
        "PROJECTION_FEEDFORWARD_ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
        "INFERENCE_ENCODER_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "INFERENCE_ENCODER_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "OUTPUT_FEEDFORWARD_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "OUTPUT_FEEDFORWARD_HIDDEN_DIM": RandomSearch.random_integer(64, 512),
        "OUTPUT_FEEDFORWARD_ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
        "OUTPUT_FEEDFORWARD_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "SEED": RandomSearch.random_integer(0, 100000),
        "GRAD_NORM": RandomSearch.random_uniform(5, 10),
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
}

NER_SEARCH = {
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 1,
        "SEED": RandomSearch.random_integer(0, 100),
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "ALPHA": RandomSearch.random_loguniform(1e-3, 1),
        "DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "ELMO_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "BERT_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "ENCODER_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "FIRST_LAYER_ONLY": 0,
        "SECOND_TO_LAST_LAYER_ONLY": 1,
        "LAST_LAYER_ONLY": 0,
        "SUM_LAST_FOUR_LAYERS": 0,
        "CONCAT_LAST_FOUR_LAYERS": 0,
        "SUM_ALL_LAYERS": 0,
        "SCALAR_MIX": 0,            
        "BATCH_SIZE": 16,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "MAX_FILTER_SIZE": RandomSearch.random_integer(3, 6),
        "NUM_FILTERS": RandomSearch.random_integer(16, 64),
        "ENCODER_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "GRAD_NORM": RandomSearch.random_uniform(5, 10),
        "CHARACTER_EMBEDDING_DIM": RandomSearch.random_integer(16, 64),
}

WORD_OVERLAP_SEARCH = {
        "EVALUATE_ON_TEST": 1,
        "ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
        "NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "HIDDEN_DIM": RandomSearch.random_integer(64, 512),
        "DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "BATCH_SIZE": 64,
        "NUM_EPOCHS": 140,
        "GRAD_NORM": RandomSearch.random_uniform(5, 10),
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "SEED": RandomSearch.random_integer(0, 100000),
}

DGEM_SEARCH = {
        "SEED": RandomSearch.random_integer(0, 100000),
        "PROJECTION_DIM": RandomSearch.random_integer(64, 300),
        "PREMISE_ENCODER_BIDIRECTIONAL":  RandomSearch.random_choice(0, 1),
        "PREMISE_ENCODER_NUM_LAYERS":  RandomSearch.random_choice(1, 2, 3),
        "PREMISE_ENCODER_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "PHRASE_PROBABILITY_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "PHRASE_PROBABILITY_HIDDEN_DIMS": RandomSearch.random_integer(64, 512),
        "PHRASE_PROBABILITY_ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
        "PHRASE_PROBABILITY_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "PHRASE_PROBABILITY_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "EDGE_PROBABILITY_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "EDGE_PROBABILITY_HIDDEN_DIMS": RandomSearch.random_integer(64, 512),
        "EDGE_PROBABILITY_ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
        "EDGE_PROBABILITY_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "EDGE_PROBABILITY_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "EDGE_EMBEDDING_SIZE": RandomSearch.random_integer(64, 300),
        "BATCH_SIZE": 16,
        "NUM_EPOCHS": 140,
        "GRAD_NORM": RandomSearch.random_uniform(5, 10),
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "CUDA_DEVICE": 0
}
BIDAF_SEARCH = {
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 1,
        "SEED": RandomSearch.random_integer(0, 100),
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "ALPHA": RandomSearch.random_loguniform(1e-3, 1),
        "DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "ELMO_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "ENCODER_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "BATCH_SIZE": 64,
        "NUM_ENCODER_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "MAX_FILTER_SIZE": RandomSearch.random_integer(3, 6),
        "NUM_FILTERS": RandomSearch.random_integer(16, 64),
        "ENCODER_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "PHRASE_LAYER_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "MODELING_LAYER_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "SPAN_END_ENCODER_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "GRAD_NORM": RandomSearch.random_uniform(5, 10),
        "CHARACTER_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "PHRASE_LAYER_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "MODELING_LAYER_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "SPAN_END_ENCODER_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "NUM_HIGHWAY_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "NUM_PHRASE_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "NUM_MODELING_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "SPAN_END_ENCODER_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "CHARACTER_EMBEDDING_DIM": RandomSearch.random_integer(16, 64),
        "BETA_1": RandomSearch.random_uniform(0.9, 1.0),
        "BETA_2": RandomSearch.random_uniform(0.9, 1.0)
}


FASTTEXT_SEARCH = {
        "label_prefix": "__label__",
        "lr": RandomSearch.random_loguniform(1e-3, 1),
        "ws": RandomSearch.random_loguniform(2, 10),
        "lr_update_rate": RandomSearch.random_integer(1, 500),
        "dim": RandomSearch.random_integer(64, 512),
        "epoch": 50,
        "thread": 20,
        "neg": RandomSearch.random_integer(1, 10),
        "word_ngrams": RandomSearch.random_choice(1, 2, 3, 4, 5),
        "loss": RandomSearch.random_choice("softmax", "ns"),
        "bucket": 20000,
        "maxn": RandomSearch.random_integer(0, 10),
        # "t": RandomSearch.random_loguniform(1e-6, 1e-1),
}

SKLEARN_LR_SEARCH = {
        "penalty": RandomSearch.random_choice("l1", "l2"),
        "C": RandomSearch.random_uniform(0, 1),
        "solver": "liblinear",
        "tol": RandomSearch.random_loguniform(10e-5, 10e-3),
        "stopwords": RandomSearch.random_choice(0, 1),
        "weight": RandomSearch.random_choice("tf", "tf-idf", "binary"),
        "ngram_range": RandomSearch.random_pair("1", "2", "3"),
        "random_state": RandomSearch.random_integer(0, 100000)
}

BIATTENTIVE_CLASSIFICATION_NETWORK_SEARCH_SST = {
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 1,
        "NUM_EPOCHS": 50,
        "SEED": RandomSearch.random_integer(0, 100),
        "THROTTLE": None,
        "GRAD_NORM": RandomSearch.random_integer(5, 10),
        "FREEZE_ELMO": 0,
        "EMBEDDING_DROPOUT": RandomSearch.random_integer(0, 5),
        "PRE_ENCODE_FEEDFORWARD_LAYERS": RandomSearch.random_integer(1, 2, 3),
        "PRE_ENCODE_FEEDFORWARD_HIDDEN_DIMS": RandomSearch.random_integer(64, 512),
        "PRE_ENCODE_FEEDFORWARD_ACTIVATION": RandomSearch.random_choice("relu", "tanh"),
        "PRE_ENCODE_FEEDFORWARD_DROPOUT": RandomSearch.random_integer(0, 5),
        "ENCODER_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "ENCODER_NUM_LAYERS":  RandomSearch.random_choice(1, 2, 3),
        "INTEGRATOR_HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "INTEGRATOR_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "INTEGRATOR_DROPOUT": RandomSearch.random_integer(0, 5),
        "ELMO_DROPOUT": RandomSearch.random_integer(0, 5),
        "OUTPUT_NUM_LAYERS": RandomSearch.random_choice(1, 2, 3),
        "OUTPUT_DIM": RandomSearch.random_integer(64, 512),
        "OUTPUT_DROPOUT": RandomSearch.random_integer(0, 5),
        "POOL_SIZES": RandomSearch.random_integer(3, 7),
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "BATCH_SIZE": 64,
        "USE_INTEGRATOR_OUTPUT_ELMO": RandomSearch.random_choice(0, 1),
}


ELMO_LSTM = {
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "EMBEDDING": "ELMO_LSTM",
        "ENCODER": "AVERAGE",
        "LEARNING_RATE": 10,
        "DROPOUT": 5,
        "BATCH_SIZE": 32,
}

ELMO_TRANSFORMER = {
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 1,
        "EMBEDDING": "ELMO_TRANSFORMER",
        "ENCODER": "AVERAGE",
        "LEARNING_RATE": 10,
        "DROPOUT": 5,
        "BATCH_SIZE": 32,
}

BERT = {
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 1,
        "EMBEDDING": "BERT",
        "ENCODER": "CLS_TOKEN",
        "LEARNING_RATE": 40,
        "DROPOUT": 0,
        "BATCH_SIZE": 32
}

GLOVE = {
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 1,
        "EMBEDDING": "GLOVE",
        "ENCODER": "AVERAGE",
        "LEARNING_RATE": 10,
        "DROPOUT": 5,
        "BATCH_SIZE": 32
}

W2V = {
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 1,
        "EMBEDDING": "W2V",
        "ENCODER": "AVERAGE",
        "LEARNING_RATE": 10,
        "DROPOUT": 5,
        "BATCH_SIZE": 32
}


CHAR_CNN = {
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 1,
        "EMBEDDING": "CHAR_CNN",
        "ENCODER": "AVERAGE",
        "LEARNING_RATE": 10,
        "DROPOUT": 5,
        "BATCH_SIZE": 32
}

CHAR_LSTM = {
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 1,
        "EMBEDDING": "CHAR_LSTM",
        "ENCODER": "AVERAGE",
        "LEARNING_RATE": 10,
        "DROPOUT": 5,
        "BATCH_SIZE": 32
}

RANDOM = {
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 1,
        "EMBEDDING": "RANDOM",
        "ENCODER": "AVERAGE",
        "LEARNING_RATE": 10,
        "DROPOUT": 5,
        "BATCH_SIZE": 32
}

ENVIRONMENTS = {
        "ELMO_LSTM": ELMO_LSTM,
        "ELMO_TRANSFORMER": ELMO_TRANSFORMER,
        "BERT": BERT,
        "GLOVE": GLOVE,
        "W2V": W2V,
        "CHAR_CNN": CHAR_CNN,
        "CHAR_LSTM": CHAR_LSTM,
        "RANDOM": RANDOM,
        "BOW_LINEAR": BOW_LINEAR,
        "SST_LSTM_GLOVE_SEARCH": SST_LSTM_GLOVE_SEARCH,
        "SST_FROZEN_ELMO_SEARCH": SST_FROZEN_ELMO_SEARCH,
        "SST_FINE_TUNED_ELMO_SEARCH": SST_FINE_TUNED_ELMO_SEARCH,
        "CLASSIFIER_LSTM_SEARCH": CLASSIFIER_LSTM_SEARCH,
        "CLASSIFIER_CNN_SEARCH": CLASSIFIER_CNN_SEARCH,
        "CLASSIFIER_BOE_SEARCH": CLASSIFIER_BOE_SEARCH,
        "NER_SEARCH": NER_SEARCH,
        "BIDAF_SEARCH": BIDAF_SEARCH,
        "DAM_SEARCH": DAM_SEARCH,
        "DGEM_SEARCH": DGEM_SEARCH,
        "ESIM_SEARCH": ESIM_SEARCH,
        "WORD_OVERLAP_SEARCH": WORD_OVERLAP_SEARCH,
        "FASTTEXT_SEARCH": FASTTEXT_SEARCH,
        "SKLEARN_LR_SEARCH": SKLEARN_LR_SEARCH,
        "BIATTENTIVE_CLASSIFICATION_NETWORK_SEARCH_SST": BIATTENTIVE_CLASSIFICATION_NETWORK_SEARCH_SST
}
