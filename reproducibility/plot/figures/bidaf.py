from reproducibility.plot.plotter import Plotter


class BIDAF(Plotter):
    def __init__(self):
        super().__init__()
        self.subplots = (1, 1)
        self.figsize = (8, 9)
        self.save_path = f"{self.root}/figures/section3_squad.pdf"
        self.config = {
            f"{self.root}/bidaf_master_1_search.tsv": {
                "linestyle": "-",
                "logx": True,
                "performance_metric": "EM",
                'duration_field': 'training_duration',
                'dataset_size_field': 'dataset_reader.sample',
                'dev_performance_field': 'best_validation_em',
                'model_field': 'embedding',
                'sep': '\t',
                'lr_field': 'trainer.optimizer.lr',
                "encoder_name": None,
                "plot_errorbar": True,
                "legend_loc": 'lower right',
                "x_axis_time": True,
                "linewidth": 3,
                "errorbar_kind": "shade",
                "errorbar_alpha": 0.075,
                "relabel_logx_scalar": [28800, 86400, 259200, 864000, 1555200, 2.592e+6],
                "reported_accuracy": {"glove":.677},
                "rename_labels": {"glove": "BIDAF"},
                "data_name": "SQuAD",
                "xlim": [28800, 3.456e+6],
                "ylim": [0.3, 0.7],
                "fontsize": 24
            },
        }

if __name__ == "__main__":
    bidaf = BIDAF()
    bidaf.plot()