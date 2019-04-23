
from environments.random_search import RandomSearch

DATA_DIR = "/home/suching/reproducibility/data/sst/"

BOW_LINEAR = {
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "EMBEDDING": "BOW_COUNTS",
        "ENCODER": None,
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "BATCH_SIZE": 32,
}


# CLASSIFIER_SEARCH = {
#         "CUDA_DEVICE": 0,
#         "USE_SPACY_TOKENIZER": 0,
#         "SEED": RandomSearch.random_integer(0, 100),
#         "DATA_DIR": DATA_DIR,
#         "THROTTLE": None,
#         "EMBEDDING": "ELMO_TRANSFORMER",
#         "ENCODER": "LSTM",
#         "HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
#         "NUM_ENCODER_LAYERS": RandomSearch.random_integer(1, 3),
#         "MAX_FILTER_SIZE": RandomSearch.random_integer(5, 10),
#         "NUM_FILTERS": RandomSearch.random_integer(64, 512),
#         "AGGREGATIONS": RandomSearch.random_choice("final_state", "maxpool", "meanpool", "attention"),
#         "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
#         "DROPOUT": RandomSearch.random_integer(0, 5),
#         "BATCH_SIZE": 64
# }

CLASSIFIER_SEARCH = {
        "LAZY_DATASET_READER": 0,
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 1,
        "NUM_EPOCHS": 50,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "USE_SPACY_TOKENIZER": 0,
        "FREEZE_EMBEDDINGS": None,
        "EMBEDDINGS": ["ELMO_LSTM"],
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
        "SEED": RandomSearch.random_integer(0, 100),
        "GRAD_CLIP": RandomSearch.random_uniform(5, 10),
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "BATCH_SIZE": 64,
        "NUM_EPOCHS": 140
}



NER_SEARCH = {
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 1,
        "SEED": RandomSearch.random_integer(0, 100),
        "LEARNING_RATE": 5e-5,
        "ALPHA": 0.0,
        "DROPOUT": 0.1,
        "ELMO_DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "BERT_DROPOUT": 0.1,
        "ENCODER_DROPOUT": 0.1,
        "FIRST_LAYER_ONLY": 0,
        "SECOND_TO_LAST_LAYER_ONLY": 1,
        "LAST_LAYER_ONLY": 0,
        "SUM_LAST_FOUR_LAYERS": 0,
        "CONCAT_LAST_FOUR_LAYERS": 0,
        "SUM_ALL_LAYERS": 0,
        "SCALAR_MIX": 0,            
        "BATCH_SIZE": 16,
        "NUM_ENCODER_LAYERS": 2,
        "MAX_FILTER_SIZE": RandomSearch.random_integer(3, 6),
        "NUM_FILTERS": RandomSearch.random_integer(16, 64),
        "ENCODER_HIDDEN_SIZE": 768,
        "GRAD_NORM": 7,
        "CHARACTER_EMBEDDING_DIM": RandomSearch.random_integer(16, 64),
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
        "CLASSIFIER_SEARCH": CLASSIFIER_SEARCH,
        "NER_SEARCH": NER_SEARCH,
        "BIDAF_SEARCH": BIDAF_SEARCH,
        "DAM_SEARCH": DAM_SEARCH,
        "BIATTENTIVE_CLASSIFICATION_NETWORK_SEARCH_SST": BIATTENTIVE_CLASSIFICATION_NETWORK_SEARCH_SST
}
