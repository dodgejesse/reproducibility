from reproducibility.plot.plotter import Plotter

class SimpleModelsSST5(Plotter):
    def __init__(self):
        super().__init__()
        self.subplots = (1, 1)
        self.figsize = (10, 10)
        self.save_path = f"{self.root}/figures/simple_models_sst5.pdf"
        self.config = {
            f"{self.root}/sst5_cnn_lr_1_search.tsv": {
                "linestyle": "-",
                "markers": [">", "o"],
                "markersize": 10,
                "logx": False,
                'sep': '\t',
                "encoder_name": None,
                'duration_field': 'training_duration',
                'dataset_size_field': 'dataset_reader.sample',
                'dev_performance_field': 'best_validation_accuracy',
                'model_field': 'model.encoder.architecture.type',
                'lr_field': 'trainer.optimizer.lr',
                "plot_errorbar": True,
                "errorbar_kind": "shade",
                "errorbar_alpha": 0.075,
                "legend_loc": 'lower left',
                "x_axis_time": False,
                "linewidth": 3,
                "data_name": "",
                "xticks_to_show": [10, 16, 20, 30, 40, 50],
                "xlim": [1, 50],
                "performance_metric": "accuracy",
                "model_order": ["LR", "CNN"],
                "fontsize": 24
            }
        }


        
if __name__ == "__main__":
    simple_models = SimpleModelsSST5()
    simple_models.plot()
