
from environments.random_search import RandomSearch

DATA_DIR = "s3://suching-dev/final-datasets/imdb/"

BOW_LINEAR = {
        "CUDA_DEVICE": -1,
        "USE_SPACY_TOKENIZER": 0,
        "SEED": RandomSearch.random_integer(0, 100),
        "DATA_DIR": DATA_DIR,
        "THROTTLE": None,
        "EMBEDDING": "BOW_COUNTS",
        "ENCODER": None,
        "LEARNING_RATE": RandomSearch.random_loguniform(1, 1000),
        "DROPOUT": RandomSearch.random_integer(0, 5),
        "BATCH_SIZE": 32,
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
        "BOW_LINEAR": BOW_LINEAR
}
