DAM_SEARCH = {
        "CUDA_DEVICE": 0,
        "EVALUATE_ON_TEST": 1,
        "PROJECTION_DIM": {
            "sampling strategy": "integer",
            "bounds": (64, 300),
        },
        "ATTEND_FEEDFORWARD_NUM_LAYERS": {
            "sampling strategy": "choice",
            "choices": [1, 2, 3]
        },
        "ATTEND_FEEDFORWARD_HIDDEN_DIMS": {
            "sampling strategy": "integer",
            "bounds": [64, 512]
        },
        "ATTEND_FEEDFORWARD_ACTIVATION": {
            "sampling strategy": "choice",
            "choices": ["relu", "tanh"]
        },
        "ATTEND_FEEDFORWARD_DROPOUT": {
            "sampling strategy": "uniform",
            "bounds": [0, 0.5]
        },
        "COMPARE_FEEDFORWARD_NUM_LAYERS": {
            "sampling strategy": "choice",
            "choices": [1, 2, 3]
        },
        "COMPARE_FEEDFORWARD_HIDDEN_DIMS": {
            "sampling strategy": "integer",
            "bounds": [64, 512]
        },
        "COMPARE_FEEDFORWARD_ACTIVATION": {
            "sampling strategy": "choice",
            "choices": ["relu", "tanh"]
        },
        "COMPARE_FEEDFORWARD_DROPOUT": {
            "sampling strategy": "uniform",
            "bounds": [0, 0.5]
        },
        "AGGREGATE_FEEDFORWARD_NUM_LAYERS": {
            "sampling strategy": "choice",
            "choices": [1, 2, 3]
        },
        "AGGREGATE_FEEDFORWARD_HIDDEN_DIMS": {
            "sampling strategy": "integer",
            "bounds": [64, 512]
        },
        "AGGREGATE_FEEDFORWARD_ACTIVATION": {
            "sampling strategy": "choice",
            "choices": ["relu", "tanh"]
        },
        "AGGREGATE_FEEDFORWARD_DROPOUT": {
            "sampling strategy": "uniform",
            "bounds": [0, 0.5]
        },
        "SEED": {
            "sampling strategy": "integer",
            "bounds": [0, 100000]
        },
        "GRAD_CLIP": {
            "sampling strategy": "uniform",
            "bounds": [5, 10]
        },
        "LEARNING_RATE": {
            "sampling strategy": "loguniform",
            "bounds": [1e-6, 1e-1]
        },
        "BATCH_SIZE": 64,
        "NUM_EPOCHS": 140
}