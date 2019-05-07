from reproducibility.plot.plotter import Plotter

class Section3_BERTonSTILTs_full(Plotter):
    def __init__(self):
        super().__init__()
        self.subplots = (2, 2)
        self.figsize = (14, 14)
        self.save_path = f"{self.root}/figures/section3_bert_on_stilts_full.pdf"
        self.config = {
            f"{self.root}/bert_large_cola_full_search.tsv": {
                "linestyle": "-",
                "logx": True,
                "performance_metric": "accuracy",
                'duration_field': 'training_duration',
                'dataset_size_field': 'dataset_reader.sample',
                'dev_performance_field': 'best_validation_accuracy',
                'model_field': 'model.encoder.architecture.type',
                'sep': ',',
                'lr_field': 'trainer.optimizer.lr',
                "encoder_name": None,
                "plot_errorbar": False,
                "legend_loc": 'lower right',
                "x_axis_time": False,
                "linewidth": 3,
                "relabel_logx_scalar": [5, 10, 15],
                "data_name": "CoLA",
                "xlim": [1, 20],
                "fontsize": 24
            },
            f"{self.root}/bert_large_mrpc_full_search.tsv": {
                "linestyle": "-",
                "logx": True,
                "performance_metric": "accuracy",
                'duration_field': 'training_duration',
                'dataset_size_field': 'dataset_reader.sample',
                'dev_performance_field': 'best_validation_accuracy',
                'model_field': 'model.encoder.architecture.type',
                'sep': ',',
                'lr_field': 'trainer.optimizer.lr',
                "encoder_name": None,
                "plot_errorbar": False,
                "legend_loc": 'lower right',
                "x_axis_time": False,
                "linewidth": 3,
                "relabel_logx_scalar": [5, 10, 15],
                "data_name": "MRPC",
                "xlim": [1, 20],
                "fontsize": 24
            },
            f"{self.root}/bert_large_rte_full_search.tsv": {
                "linestyle": "-",
                "logx": True,
                "performance_metric": "accuracy",
                'duration_field': 'training_duration',
                'dataset_size_field': 'dataset_reader.sample',
                'dev_performance_field': 'best_validation_accuracy',
                'model_field': 'model.encoder.architecture.type',
                'sep': ',',
                'lr_field': 'trainer.optimizer.lr',
                "encoder_name": None,
                "plot_errorbar": False,
                "legend_loc": 'lower right',
                "x_axis_time": False,
                "linewidth": 3,
                "relabel_logx_scalar": [5, 10, 15],
                "data_name": "RTE",
                "xlim": [1, 20],
                "fontsize": 24
            },
            f"{self.root}/bert_large_stsb_full_search.tsv": {
                "linestyle": "-",
                "logx": True,
                "performance_metric": "accuracy",
                'duration_field': 'training_duration',
                'dataset_size_field': 'dataset_reader.sample',
                'dev_performance_field': 'best_validation_accuracy',
                'model_field': 'model.encoder.architecture.type',
                'sep': ',',
                'lr_field': 'trainer.optimizer.lr',
                "encoder_name": None,
                "plot_errorbar": False,
                "legend_loc": 'lower right',
                "x_axis_time": False,
                "linewidth": 3,
                "relabel_logx_scalar": [5, 10, 15],
                "data_name": "STS",
                "xlim": [1, 20],
                "fontsize": 24
            }
        }



if __name__ == "__main__":
    bert_on_stilts = Section3_BERTonSTILTs_full()
    bert_on_stilts.plot()