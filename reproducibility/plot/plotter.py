import argparse
import collections
import os
import sys
import scipy
from typing import Dict, List, Tuple
from labellines import labelLine, labelLines

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import datetime

sns.set_style("white")


class Plotter:
    def __init__(self):
        self.root = "/Users/suching/Github/reproducibility/final_results"
        if not os.path.exists(f"{self.root}/figures"):
            os.mkdir(f"{self.root}/figures")
    

    @staticmethod
    def _cdf_with_replacement(i,n,N):
        return (i/N)**n

    @staticmethod
    def _cdf_without_replacement(i,n,N):
        return scipy.special.comb(i,n) / scipy.special.comb(N,n)

    @staticmethod
    def _compute_variance(N, cur_data, expected_max_cond_n, pdfs):
        """
        this computes the standard error of the max.
        this is what the std dev of the bootstrap estimates of the mean of the max converges to, as
        is stated in the last sentence of the summary on page 10 of http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf
        uses equation 
        """
        variance_of_max_cond_n = []
        for n in range(N):
            # for a given n, estimate variance with \sum(p(x) * (x-mu)^2), where mu is \sum(p(x) * x).
            cur_var = 0
            for i in range(N):
                cur_var += (cur_data[i] - expected_max_cond_n[n])**2 * pdfs[n][i]
            cur_var = np.sqrt(cur_var)
            variance_of_max_cond_n.append(cur_var)
        return variance_of_max_cond_n
        

    # this implementation assumes sampling with replacement for computing the empirical cdf
    def samplemax(self, validation_performance, with_replacement=True):
        validation_performance = list(validation_performance)
        validation_performance.sort()
        N = len(validation_performance)
        pdfs = []
        for n in range(1,N+1):
            # the CDF of the max
            F_Y_of_y = []
            for i in range(1,N+1):
                if with_replacement:
                    F_Y_of_y.append(self._cdf_with_replacement(i,n,N))
                else:
                    F_Y_of_y.append(self._cdf_without_replacement(i,n,N))

            f_Y_of_y = []
            cur_cdf_val = 0
            for i in range(len(F_Y_of_y)):
                f_Y_of_y.append(F_Y_of_y[i] - cur_cdf_val)
                cur_cdf_val = F_Y_of_y[i]
            
            pdfs.append(f_Y_of_y)

        expected_max_cond_n = []
        for n in range(N):
            # for a given n, estimate expected value with \sum(x * p(x)), where p(x) is prob x is max.
            cur_expected = 0
            for i in range(N):
                cur_expected += validation_performance[i] * pdfs[n][i]
            expected_max_cond_n.append(cur_expected)


        var_of_max_cond_n = self._compute_variance(N, validation_performance, expected_max_cond_n, pdfs)

        return {"mean":expected_max_cond_n, "var":var_of_max_cond_n, "max": np.max(validation_performance)}

    @staticmethod
    def td_format(td_object):
        seconds = int(td_object.total_seconds())
        periods = [
            ('yr',        60*60*24*365),
            ('mo',       60*60*24*30),
            ('d',         60*60*24),
            ('h',        60*60),
            ('min',      60),
            ('sec',      1)
        ]
        strings=[]
        for period_name, period_seconds in periods:
            if seconds > period_seconds:
                period_value , seconds = divmod(seconds, period_seconds)
                has_s = 's' if period_value > 1 and period_name not in ['min', 'sec', 'd', 'h'] else ''
                strings.append("%s%s%s" % (period_value, period_name, has_s))
        res = ", ".join(strings)
        if res == '60min':
            res = '1h'
        elif res == '24h':
            res = '1d'
        elif res == '30d':
            res = '1mo'
        return res

    def _one_plot(self,
                data: pd.DataFrame,
                avg_time: pd.DataFrame,
                data_size: int,
                cur_ax: matplotlib.axis,
                data_name: str = "SST5",
                linestyle: str = "-",
                linewidth: int = 3,
                logx: bool = False,
                plot_errorbar: bool = False,
                errorbar_kind: str = 'shade',
                errorbar_alpha: float = 0.1,
                x_axis_time: bool = False,
                legend_loc: str = 'lower right',
                markers: List[str] = None,
                markersize : int = None,
                relabel_logx_scalar: List[int] = None,
                rename_labels: Dict[str, str] = None,
                reported_accuracy: List[float] = None,
                encoder_name: str = None,
                show_xticks: bool = False,
                xticks_to_show: List[int] = None,
                fontsize: int = 16,
                xlim: List[int] = None,
                ylim: List[int] = None,
                model_order: List[str] = None,
                performance_metric: str = "accuracy",
                rot: int = 0,
                line_colors: List[str] = ["#8c564b", '#1f77b4', '#ff7f0e', '#17becf'],
                errorbar_colors: List[str] = ['#B22222', "#089FFF", "#228B22"]):
    
        cur_ax.set_title(data_name, fontsize=fontsize)
        if model_order:
            models = model_order
        else:
            models = data.index.levels[0].tolist()
            models.sort()
        max_first_point = 0
        cur_ax.set_ylabel("Expected validation " + performance_metric, fontsize=fontsize)
        
        if x_axis_time:
            cur_ax.set_xlabel("Training duration",fontsize=fontsize)
        else:
            cur_ax.set_xlabel("Hyperparameter assignments",fontsize=fontsize)
        
        if logx:
            cur_ax.set_xscale('log')
        
        for ix, model in enumerate(models):
            means = data[model]['mean']
            vars = data[model]['var']
            max_acc = data[model]['max']
            if x_axis_time:
                x_axis = [avg_time[model] * (i+1) for i in range(len(means))]
            else:
                x_axis = [i+1 for i in range(len(means))]

            if rename_labels:
                model_name = rename_labels.get(model, model)
            else:
                model_name = model
            if reported_accuracy:
                cur_ax.plot([0, 100],
                            [reported_accuracy[model],
                            reported_accuracy[model]],
                            linestyle='--',
                            linewidth=linewidth,
                            color=line_colors[ix])
                plt.text(100 - 3,
                        reported_accuracy[model] - 0.004,
                        f'reported {model_name} {performance_metric}',
                        ha='right',
                        style='italic',
                        fontsize=fontsize-5,
                        color=line_colors[ix])

                # cur_ax.plot([0, 6.912e+6],
                #             [reported_accuracy[model],
                #             reported_accuracy[model]],
                #             linestyle='--',
                #             linewidth=linewidth,
                #             color=line_colors[ix])
                # plt.text(6.912e+6-3600000,
                #         reported_accuracy[model] + 0.01,
                #         f'reported {model_name} {performance_metric}',
                #         ha='right',
                #         style='italic',
                #         fontsize=fontsize-5,
                #         color=line_colors[ix])

            if encoder_name:
                model_name = encoder_name + " " + model_name

            if plot_errorbar:
                if errorbar_kind == 'shade':
                    minus_vars = np.array(means)-np.array(vars)
                    plus_vars = [x + y if (x + y) <= max_acc else max_acc for x,y in zip(means, vars)]
                    plt.fill_between(x_axis,
                                     minus_vars,
                                     plus_vars,
                                     alpha=errorbar_alpha,
                                     facecolor=errorbar_colors[ix])
                else:
                    line = cur_ax.errorbar(x_axis,
                                    means,
                                    yerr=vars,
                                    label=model_name,
                                    linestyle=linestyle,
                                    linewidth=linewidth,
                                    color=line_colors[ix])
            line = cur_ax.plot(x_axis,
                                means,
                                marker=markers[ix],
                                markevery=0.1,
                                markersize=markersize,
                                label=model_name,
                                linestyle=linestyle,
                                linewidth=linewidth,
                                color=line_colors[ix])
            # labelLine(line[0], x=inline_label_loc[ix],)
        left, right = cur_ax.get_xlim()
        if ylim:
            cur_ax.set_ylim(ylim)
        if xlim:
            cur_ax.set_xlim(xlim)
            # cur_ax.xaxis.set_ticks(np.arange(xlim[0], xlim[1]+5, 10))
        for tick in cur_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
        for tick in cur_ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
        plt.locator_params(axis='y', nbins=10)
        if relabel_logx_scalar:
            for axis in [cur_ax.xaxis]:
                axis.set_ticks(relabel_logx_scalar)
                axis.set_major_formatter(ScalarFormatter())
        plt.xticks(rotation=rot)
        
        if show_xticks:
            cur_ax.tick_params(which="both", bottom=True)
        if xticks_to_show:
            for axis in [cur_ax.xaxis]:
                axis.set_ticks(xticks_to_show)
        if x_axis_time:
            def timeTicks(x, pos):                                                                                                                                                                                                                                                         
                d = datetime.timedelta(seconds=float(x))
                d = self.td_format(d)
                return str(d)                                                                                                                                                                                                                                                          
            formatter = matplotlib.ticker.FuncFormatter(timeTicks)                                                                                                                                                                                                                         
            cur_ax.xaxis.set_major_formatter(formatter)
        cur_ax.legend(loc=legend_loc, fontsize=fontsize)
        
        plt.tight_layout()

    def plot(self):
        expected_max_performance_data = {}
        average_times = {}
        f, axes = plt.subplots(self.subplots[0], self.subplots[1], figsize=self.figsize)
        if self.subplots != (1, 1):
            axes_iter = zip(self.config.items(), np.ndenumerate(axes))
        else:
            axes_iter = zip(self.config.items(), enumerate([axes]))
        
        for ((data_file, configuration), (index, _)) in axes_iter:
            sep = self.config[data_file].pop('sep')
            duration_field = self.config[data_file].pop('duration_field')
            model_field = self.config[data_file].pop('model_field')
            dataset_size_field = self.config[data_file].pop('dataset_size_field')
            dev_performance_field = self.config[data_file].pop('dev_performance_field')
            lr_field = self.config[data_file].pop('lr_field')
            master = pd.read_csv(data_file, sep=sep)
            data_sizes = master[dataset_size_field].unique()
            for data_size in data_sizes:
                df = master.loc[master['dataset_reader.sample'] == data_size]
                avg_time = df.groupby(model_field)[duration_field].mean()
                perf = {group_name: group[dev_performance_field].tolist() for group_name, group in df.groupby(model_field)}
                sample_maxes = df.groupby(model_field)[dev_performance_field].apply(self.samplemax)
                expected_max_performance_data[data_file] = {data_size: sample_maxes}
                average_times[data_file] = {data_size: avg_time}
                if self.subplots == (1,1):
                    axis = axes
                elif self.subplots[1] > 1:
                    axis = axes[index[0], index[1]]
                else:
                    axis = axes[index[0]]
                self._one_plot(sample_maxes,
                            avg_time,
                            data_size,
                            axis,
                            **self.config[data_file])
        print("saving to {}".format(self.save_path))
        plt.savefig(self.save_path, dpi=300)
