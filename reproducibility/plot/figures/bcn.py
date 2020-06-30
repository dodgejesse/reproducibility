from reproducibility.plot.plotter import Plotter


class BCN(Plotter):
    def __init__(self):
        super().__init__()
        self.subplots = (1, 1)
        self.figsize = (12, 10)
        self.save_path = f"{self.root}/figures/bcn.pdf"
        self.config = {
            f"{self.root}/sst2_biattentive_classifier_search.tsv": {
                "linestyle": "-",
                "logx": True,
                'sep': '\t',
                "markers": [">", "o", "s"],
                "markersize": 10,
                'duration_field': 'training_duration',
                'dataset_size_field': 'dataset_reader.sample',
                'dev_performance_field': 'best_validation_accuracy',
                'model_field': 'embedding',
                'lr_field': 'trainer.optimizer.lr',
                "encoder_name": None,
                "plot_errorbar": False,
                "show_xticks": True,
                "legend_loc": 'lower right',
                "x_axis_time": True,
                "linewidth": 3,
                "relabel_logx_scalar": [1800, 3600, 21600, 86400, 259200, 864000],
                "rot": 0,
                "data_name": "SST (binary)",
                "rename_labels": {'elmo frozen': "GloVe + ELMo (FR)", 'elmo fine-tuned': "GloVe + ELMo (FT)", 'glove': "GloVe"},
                "performance_metric": "accuracy",
                "fontsize": 24
            },
        }

if __name__ == "__main__":
    bcn = BCN()
    bcn.plot()