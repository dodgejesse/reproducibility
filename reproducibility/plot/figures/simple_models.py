from reproducibility.plot.plotter import Plotter

class SimpleModels(Plotter):
    def __init__(self):
        super().__init__()
        self.subplots = (2, 1)
        self.figsize = (11, 10)
        self.save_path = f"{self.root}/figures/simple_models.pdf"
        self.config = {
            f"{self.root}/sst5_cnn_lr_search.tsv": {
                "linestyle": "-",
                "logx": False,
                'sep': '\t',
                "encoder_name": None,
                'duration_field': 'training_duration',
                'dataset_size_field': 'dataset_reader.sample',
                'dev_performance_field': 'best_validation_accuracy',
                'model_field': 'model.encoder.architecture.type',
                'lr_field': 'trainer.optimizer.lr',
                "plot_errorbar": False,
                "legend_loc": 'lower right',
                "x_axis_time": False,
                "linewidth": 3,
                "data_name": "SST5",
                "xlim": [0, 50],
                "performance_metric": "accuracy",
                "model_order": ["logistic regression", "cnn"],
                "fontsize": 24
            },
            f"{self.root}/imdb_final_search.tsv": {
                "linestyle": "-",
                "logx": True,
                'sep': '\t',
                'duration_field': 'training_duration',
                'dataset_size_field': 'dataset_reader.sample',
                'dev_performance_field': 'best_validation_accuracy',
                'model_field': 'model.encoder.architecture.type',
                'lr_field': 'trainer.optimizer.lr',
                "encoder_name": None,
                "plot_errorbar": False,
                "legend_loc": 'lower right',
                "data_name": "IMDB",
                "performance_metric": "accuracy",
                "linewidth": 3,
                "model_order": ["logistic regression", "lstm"],
                "x_axis_time": True,
                "fontsize": 24
            }
        }


        
if __name__ == "__main__":
    simple_models = SimpleModels()
    simple_models.plot()
