from reproducibility.plot.plotter import Plotter


class SciTail(Plotter):
    def __init__(self):
        super().__init__()
        self.subplots = (1, 1)
        self.figsize = (10, 19)
        self.save_path = f"{self.root}/figures/section3_scitail.pdf"
        self.config = {
            f"{self.root}/scitail_3_search.tsv": {
                "linestyle": "-",
                "logx": True,
                "markers": [">", "o", "X", "s"],
                "markersize": 10,
                "encoder_name": None,
                'sep': '\t',
                "performance_metric": "accuracy",
                'duration_field': 'training_duration',
                'dataset_size_field': 'dataset_reader.sample',
                'dev_performance_field': 'best_validation_accuracy',
                'model_field': 'model.encoder.architecture.type',
                'lr_field': 'trainer.optimizer.lr',
                "plot_errorbar": False,
                "legend_loc": 'upper left',
                "x_axis_time": False,
                "linewidth": 3,
                "relabel_logx_scalar": [5, 10, 50, 100],
                "reported_accuracy": {"word overlap":.65, "dam": .754, "esim": 0.705, "DGEM": 0.796},
                "rename_labels": {'word overlap': "n-gram baseline", 'dam': "DAM", 'esim': "ESIM"},
                "data_name": "SciTail",
                "xlim": [1, 100],
                "fontsize": 24
            },
        }


if __name__ == "__main__":
    scitail = SciTail()
    scitail.plot()