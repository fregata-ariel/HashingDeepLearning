from __future__ import annotations


class config:
    data_path_train: str = "../dataset/Amazon/amazon_train.txt"
    data_path_test: str = "../dataset/Amazon/amazon_test.txt"
    GPUs: str = "0"
    num_threads: int = 44
    lr: float = 0.0001

    feature_dim: int = 135909
    n_classes: int = 670091
    n_train: int = 490449
    n_test: int = 153025
    n_epochs: int = 2
    batch_size: int = 128
    hidden_dim: int = 128

    log_file: str = "log_amz_ss"
    n_samples: int = n_classes // 10
    max_label: int = 1
