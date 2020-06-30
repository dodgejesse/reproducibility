from reproducibility.plot.plotter import Plotter

class SimpleModelsIMDB(Plotter):
    def __init__(self):
        super().__init__()
        self.subplots = (1, 1)
        self.figsize = (10, 10)
        self.save_path = f"{self.root}/figures/simple_models_imdb.pdf"
        self.config = {
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
                "plot_errorbar": True,
                "errorbar_kind": "shade",
                "errorbar_alpha": 0.075,
                "show_xticks": True,
                "legend_loc": 'lower right',
                "data_name": "IMDB",
                "performance_metric": "accuracy",
                "relabel_logx_scalar": [60, 900, 3600, 18000, 86400, 432000],
                "rot": 0,
                "linewidth": 3,
                "model_order": ["logistic regression", "lstm"],
                "x_axis_time": True,
                "fontsize": 24
            }
        }


        
if __name__ == "__main__":
    simple_models = SimpleModelsIMDB()
    simple_models.plot()
