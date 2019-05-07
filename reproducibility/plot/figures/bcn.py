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
                'duration_field': 'training_duration',
                'dataset_size_field': 'dataset_reader.sample',
                'dev_performance_field': 'best_validation_accuracy',
                'model_field': 'embedding',
                'lr_field': 'trainer.optimizer.lr',
                "encoder_name": None,
                "plot_errorbar": False,
                "legend_loc": 'lower right',
                "x_axis_time": True,
                "linewidth": 3,
                "data_name": "SST2",
                "performance_metric": "accuracy",
                "fontsize": 24
            },
        }

if __name__ == "__main__":
    bcn = BCN()
    bcn.plot()