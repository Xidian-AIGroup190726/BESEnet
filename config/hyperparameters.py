class ExperimentHyperParameters:
    load_epoch = 0
    run_identifier = "2026_04_17_2"
    evaluate_epoch: int = 0
    load: str = None
    attention_residual_factor = 2
    beta = 0.2

    save_interval: int = 20
    epochs: int = 150
    dataset_name = "whu"
    random_seed = 42
    warm_up_step = 500

    save_dir = rf".\run\{dataset_name}\{run_identifier}/"
    log_file = r"trainValLog.txt"

    batch_size: int = 8
    inference_ratio = 2
    learning_rate: float = 2e-4
    factor = 0.1
    patience = 12
    weight_decay: float = 1e-3
    amp: bool = True
    max_norm: float = 20

    save_checkpoint: bool = True
    save_best_model: bool = True
    noise_p: float = 0.3
    dropout_p: float = 0.1
    patch_size: int = 256
    log_path = "./log_feature/"

    def state_dict(self):
        return {
            key: getattr(self, key)
            for key in ExperimentHyperParameters.__dict__
            if not key.startswith("_") and not callable(getattr(self, key))
        }


experiment_parameters = ExperimentHyperParameters()
