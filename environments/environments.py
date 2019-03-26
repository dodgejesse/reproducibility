
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


CLASSIFIER_SEARCH = {
        "CUDA_DEVICE": 0,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "EMBEDDING": RandomSearch.random_choice("BOW_COUNTS", "GLOVE", "GLOVE", "GLOVE"),
        "ENCODER": RandomSearch.random_choice("LSTM", "CNN", "AVERAGE"),
        "HIDDEN_SIZE": RandomSearch.random_integer(64, 512),
        "NUM_ENCODER_LAYERS": RandomSearch.random_integer(1, 3),
        "MAX_FILTER_SIZE": RandomSearch.random_integer(5, 10),
        "NUM_FILTERS": RandomSearch.random_integer(64, 512),
        "AGGREGATIONS": RandomSearch.random_choice("final_state", "maxpool", "meanpool", "attention"),
        "LEARNING_RATE": RandomSearch.random_loguniform(1e-6, 1e-1),
        "DROPOUT": RandomSearch.random_uniform(0, 0.5),
        "BATCH_SIZE": 32
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
        "CLASSIFIER_SEARCH": CLASSIFIER_SEARCH
}
