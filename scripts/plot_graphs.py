import sys
from scripts import samplemax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import numpy as np
import argparse
import seaborn as sns
sns.set_style("white")
linestyle = ['-', '--']
# data_name = collections.OrderedDict({"imdb_final":"IMDB"})
# data_name = collections.OrderedDict({"sst_cnn_lr":"SST5", "ag_1":"AG"})
# data_name = collections.OrderedDict({"sst_lstm_updated":"SST LSTM", "sst2_biattentive_classifier":"SST biattentive"})
# data_name = collections.OrderedDict({"scitail":"SCITAIL"})
# data_name = collections.OrderedDict({"ag_1":"AG"})
# data_name = collections.OrderedDict({"imdb_final":"IMDB", "ag_1":"AG"})
# data_name = collections.OrderedDict({"sst5_cnn_lr":"SST5", "imdb_final":"IMDB"})
# data_name = collections.OrderedDict({"sst_lstm_updated":"SST LSTM", "sst2_biattentive_classifier":"SST biattentive"})


def plot_bert_on_stilts_full(plot_errorbar, x_axis_time):
    data_name = collections.OrderedDict({"bert_large_cola_full":"CoLA", 
                                         "bert_large_mrpc_full": "MRPC",
                                         "bert_large_rte_full": "RTE",
                                         "bert_large_stsb_full": "STS-B",
                                         })

    x_axis_time = False
    data, avg_time = get_data(data_name)
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0

    f, axes = plt.subplots(2, 2, figsize=(14, 14))

    for (fn, name), ((x,y), _) in zip(data_name.items(), np.ndenumerate(axes)):

        one_plot(data[fn][10000],
                 avg_time[fn][10000],
                 10000,
                 axes[x,y],
                 name + " Full",
                 0,
                 xlim=[0, 20],
                 classifiers=['BERT Large', 'BERT Large MNLI'],
                 plot_errorbar=False,
                 legend_loc='lower right',
                 logx=False,
                 fontsize=24,
                 x_axis_time=x_axis_time)

    

    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)


def plot_bert_on_stilts_5k(plot_errorbar, x_axis_time):
    data_name = collections.OrderedDict({"cola_5k":"CoLA", 
                                         "mrpc_5k": "MRPC",
                                         "rte_5k": "RTE",
                                         "stsb_5k": "STS-B",
                                         "mnli_5k": "MNLI",
                                         "qnli_5k": "QNLI",
                                         "wnli_5k": "WNLI",
                                         "qqp_5k": "QQP",
                                         "sst2_5k": "SST2",
                                         })
    x_axis_time = False
    data, avg_time = get_data(data_name)
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0

    f, axes = plt.subplots(3, 3, figsize=(20, 20))

    for (fn, name), ((x,y), _) in zip(data_name.items(), np.ndenumerate(axes)):

        one_plot(data[fn][10000],
                 avg_time[fn][10000],
                 10000,
                 axes[x,y],
                 name + " 5K",
                 0,
                 xlim=[0, 20],
                 classifiers=['BERT Large', 'BERT Large MNLI'],
                 plot_errorbar=False,
                 legend_loc='lower right',
                 logx=False,
                 fontsize=24,
                 x_axis_time=x_axis_time)

    

    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)



def plot_bert_on_stilts_1k(plot_errorbar, x_axis_time):
    data_name = collections.OrderedDict({"cola_1k":"CoLA", 
                                         "mrpc_1k": "MRPC",
                                         "rte_1k": "RTE",
                                         "stsb_1k": "STS-B",
                                         "mnli_1k": "MNLI",
                                         "qnli_1k": "QNLI",
                                         "wnli_1k": "WNLI",
                                         "qqp_1k": "QQP",
                                         "sst2_1k": "SST2",


                                         })

    x_axis_time = False
    data, avg_time = get_data(data_name)
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0

    f, axes = plt.subplots(3, 3, figsize=(20, 20))

    for (fn, name), ((x,y), _) in zip(data_name.items(), np.ndenumerate(axes)):

        one_plot(data[fn][10000],
                 avg_time[fn][10000],
                 10000,
                 axes[x,y],
                 name + " 1K",
                 0,
                 xlim=[0, 20],
                 classifiers=['BERT Large', 'BERT Large MNLI'],
                 plot_errorbar=False,
                 legend_loc='lower right',
                 logx=False,
                 fontsize=24,
                 x_axis_time=x_axis_time)

    # one_plot(data["bert_large_mrpc_1k"][10000],
    #          avg_time["bert_large_mrpc_1k"][10000],
    #          10000,
    #          axes[0,0],
    #          "MRPC",
    #          0,
    #          xlim=[0, 20],
    #          classifiers=['BERT', 'BERT on STILTs'],
    #          plot_errorbar=False,
    #          legend_loc='lower right',
    #          logx=False,
    #          fontsize=24,
    #          x_axis_time=x_axis_time)
    
    # one_plot(data["bert_large_rte_1k"][10000],
    #          avg_time["bert_large_rte_1k"][10000],
    #          10000,
    #          axes[0,1],
    #          "RTE",
    #          0,
    #          xlim=[0, 20],
    #          classifiers=['BERT', 'BERT on STILTs'],
    #          plot_errorbar=False,
    #          legend_loc='lower right',
    #          logx=False,
    #          fontsize=24,
    #          x_axis_time=x_axis_time)
    
    # one_plot(data["bert_large_stsb_1k"][10000],
    #          avg_time["bert_large_stsb_1k"][10000],
    #          10000,
    #          axes[1,0],
    #          "STS-B",
    #          0,
    #          xlim=[0, 20],
    #          classifiers=['BERT', 'BERT on STILTs'],
    #          plot_errorbar=False,
    #          legend_loc='lower right',
    #          logx=False,
    #          fontsize=24,
    #          x_axis_time=x_axis_time)


    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)


def plot_bert_on_stilts_1k(plot_errorbar, x_axis_time):
    data_name = collections.OrderedDict({"cola_1k":"CoLA", 
                                         "mrpc_1k": "MRPC",
                                         "rte_1k": "RTE",
                                         "stsb_1k": "STS-B",
                                         "mnli_1k": "MNLI",
                                         "qnli_1k": "QNLI",
                                         "wnli_1k": "WNLI",
                                         "qqp_1k": "QQP",
                                         "sst2_1k": "SST2",


                                         })

    x_axis_time = False
    data, avg_time = get_data(data_name)
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0

    f, axes = plt.subplots(3, 3, figsize=(20, 20))

    for (fn, name), ((x,y), _) in zip(data_name.items(), np.ndenumerate(axes)):

        one_plot(data[fn][10000],
                 avg_time[fn][10000],
                 10000,
                 axes[x,y],
                 name + " 1K",
                 0,
                 xlim=[0, 20],
                 classifiers=['BERT Large', 'BERT Large MNLI'],
                 plot_errorbar=False,
                 legend_loc='lower right',
                 logx=False,
                 fontsize=24,
                 x_axis_time=x_axis_time)

    # one_plot(data["bert_large_mrpc_1k"][10000],
    #          avg_time["bert_large_mrpc_1k"][10000],
    #          10000,
    #          axes[0,0],
    #          "MRPC",
    #          0,
    #          xlim=[0, 20],
    #          classifiers=['BERT', 'BERT on STILTs'],
    #          plot_errorbar=False,
    #          legend_loc='lower right',
    #          logx=False,
    #          fontsize=24,
    #          x_axis_time=x_axis_time)
    
    # one_plot(data["bert_large_rte_1k"][10000],
    #          avg_time["bert_large_rte_1k"][10000],
    #          10000,
    #          axes[0,1],
    #          "RTE",
    #          0,
    #          xlim=[0, 20],
    #          classifiers=['BERT', 'BERT on STILTs'],
    #          plot_errorbar=False,
    #          legend_loc='lower right',
    #          logx=False,
    #          fontsize=24,
    #          x_axis_time=x_axis_time)
    
    # one_plot(data["bert_large_stsb_1k"][10000],
    #          avg_time["bert_large_stsb_1k"][10000],
    #          10000,
    #          axes[1,0],
    #          "STS-B",
    #          0,
    #          xlim=[0, 20],
    #          classifiers=['BERT', 'BERT on STILTs'],
    #          plot_errorbar=False,
    #          legend_loc='lower right',
    #          logx=False,
    #          fontsize=24,
    #          x_axis_time=x_axis_time)


    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)


def plot_section1(plot_errorbar, x_axis_time):
    data_name = collections.OrderedDict({"sst5_cnn_lr_1":"SST5"})

    x_axis_time = False
    data, avg_time = get_data(data_name)
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0

    f, axes = plt.subplots(2, 1, figsize=(10,14))


    one_plot(data["sst5_cnn_lr_1"][8544],
             avg_time["sst5_cnn_lr_1"][8544],
             8544,
             axes[0],
             "SST5",
             0,
             xlim=[0, 50],
             classifiers=['LR', 'CNN'],
             plot_errorbar=False,
             legend_loc='lower right',
             logx=False,
             fontsize=24,
             x_axis_time=x_axis_time)

    # one_plot(data["ag_1"][115000],
    #          avg_time["ag_1"][115000],
    #          115000,
    #          axes[0],
    #          "AG News",
    #          0,
    #          classifiers=['logistic regression', 'cnn'],
    #          xlim=[0, 20],
    #          logx=False,
    #          plot_errorbar=plot_errorbar,
    #          legend_loc='lower right',
    #          x_axis_time=x_axis_time)

    # one_plot(data["imdb_final"][20000],
    #          avg_time["imdb_final"][20000],
    #          20000,
    #          axes[1],
    #          "IMDB",
    #          0,
    #          logx=True,
    #          classifiers=['logistic regression', 'lstm'],
    #          plot_errorbar=plot_errorbar,
    #          legend_loc='lower right',
    #          fontsize=24,
    #          x_axis_time=True)

    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)

    # for data_size in data_sizes:
    #     counter += 1

    #     cur_ax = fig.add_subplot(sqrt_num_plots,sqrt_num_plots,counter)
    #     # cur_ax.set_xscale('log')

    #     experiment_counter = 0
    #     for cur_data in data_name:
    #         try:
    #             one_plot(data[cur_data][data_size],
    #                     avg_time[cur_data][data_size],
    #                     data_size,
    #                     cur_ax,
    #                     data_name[cur_data],
    #                     experiment_counter,
    #                     plot_errorbar=plot_errorbar,
    #                     x_axis_time=x_axis_time)
    #             experiment_counter += 1
    #         except:
    #             continue

    # classifiers = get_classifiers(data[list(data.keys())[0]])
    # save_plot(data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)


def plot_section2(plot_errorbar, x_axis_time):
    data_name = collections.OrderedDict({"sst2_lstm_original":"SST LSTM", "sst2_biattentive_classifier":"SST biattentive"})

    x_axis_time = False

    data, avg_time = get_data(data_name)
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0

    f, axes = plt.subplots(1, 1, figsize=(6,5))

    one_plot(data["sst2_lstm_original"][6919],
             avg_time["sst2_lstm_original"][6919],
             6919,
             axes,
             "SST2",
             1,
             classifiers=['glove', 'elmo frozen', 'elmo fine-tuned'],
             plot_errorbar=False,
             encoder_name="LSTM",
             legend_loc='lower right',
             logx=True,
             x_axis_time=True)

    # one_plot(data["ag_1"][115000],
    #          avg_time["ag_1"][115000],
    #          115000,
    #          axes[0],
    #          "AG News",
    #          0,
    #          classifiers=['logistic regression', 'cnn'],
    #          xlim=[0, 20],
    #          logx=False,
    #          plot_errorbar=plot_errorbar,
    #          legend_loc='lower right',
    #          x_axis_time=x_axis_time)

    one_plot(data["sst2_biattentive_classifier"][6919],
             avg_time["sst2_biattentive_classifier"][6919],
             6919,
             axes,
             "SST2",
             0,
             logx=True,
             encoder_name="BCN",
             classifiers=['glove', 'elmo frozen', 'elmo fine-tuned'],
             plot_errorbar=plot_errorbar,
             legend_loc='lower right',
             x_axis_time=True)

    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)



def plot_section2_lstms(plot_errorbar, x_axis_time):
    data_name = collections.OrderedDict({"sst2_lstm_original":"SST LSTM+sentences", "sst_lstm_updated":"SST LSTM+sentences+subtrees"})

    x_axis_time = False

    data, avg_time = get_data(data_name)
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0

    f, axes = plt.subplots(1, 1, figsize=(12,10))

    one_plot(data["sst2_lstm_original"][6919],
             avg_time["sst2_lstm_original"][6919],
             6919,
             axes,
             "SST2",
             1,
             classifiers=['glove', 'elmo frozen', 'elmo fine-tuned'],
             plot_errorbar=False,
             encoder_name="",
             legend_loc='lower right',
             logx=True,
             x_axis_time=True,
             fontsize=24)

    # one_plot(data["ag_1"][115000],
    #          avg_time["ag_1"][115000],
    #          115000,
    #          axes[0],
    #          "AG News",
    #          0,
    #          classifiers=['logistic regression', 'cnn'],
    #          xlim=[0, 20],
    #          logx=False,
    #          plot_errorbar=plot_errorbar,
    #          legend_loc='lower right',
    #          x_axis_time=x_axis_time)

    one_plot(data["sst_lstm_updated"][6919],
             avg_time["sst_lstm_updated"][6919],
             6919,
             axes,
             "SST2",
             0,
             logx=True,
             encoder_name="",
             classifiers=['glove', 'elmo frozen', 'elmo fine-tuned'],
             plot_errorbar=plot_errorbar,
             legend_loc='lower right',
             x_axis_time=True,
             fontsize=24)

    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)

def plot_section3_bcn(plot_errorbar, x_axis_time):
    data_name = collections.OrderedDict({"sst2_biattentive_classifier":"BCN"})

    x_axis_time = False

    data, avg_time = get_data(data_name)
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0

    f, axes = plt.subplots(1, 1, figsize=(12,10))

    one_plot(data["sst2_biattentive_classifier"][6919],
             avg_time["sst2_biattentive_classifier"][6919],
             6919,
             axes,
             "SST2",
             0,
             classifiers=['glove', 'elmo frozen', 'elmo fine-tuned'],
             plot_errorbar=False,
             encoder_name="",
             legend_loc='lower right',
             logx=True,
             x_axis_time=True,
             fontsize=24)

    # one_plot(data["ag_1"][115000],
    #          avg_time["ag_1"][115000],
    #          115000,
    #          axes[0],
    #          "AG News",
    #          0,
    #          classifiers=['logistic regression', 'cnn'],
    #          xlim=[0, 20],
    #          logx=False,
    #          plot_errorbar=plot_errorbar,
    #          legend_loc='lower right',
    #          x_axis_time=x_axis_time)

    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)



def plot_section2_ner(plot_errorbar, x_axis_time):
    data_name = collections.OrderedDict({"ner":"NER"})

    x_axis_time = False

    data, avg_time = get_data(data_name)
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0

    f, axes = plt.subplots(1, 1, figsize=(6,5))

    one_plot(data["ner"][10000],
             avg_time["ner"][10000],
             10000,
             axes,
             "NER",
             0,
             classifiers=['glove', 'elmo frozen', 'elmo fine-tuned'],
             plot_errorbar=False,
             encoder_name="LSTM-CRF",
             legend_loc='lower right',
             logx=True,
             x_axis_time=True)

    # one_plot(data["ag_1"][115000],
    #          avg_time["ag_1"][115000],
    #          115000,
    #          axes[0],
    #          "AG News",
    #          0,
    #          classifiers=['logistic regression', 'cnn'],
    #          xlim=[0, 20],
    #          logx=False,
    #          plot_errorbar=plot_errorbar,
    #          legend_loc='lower right',
    #          x_axis_time=x_axis_time)

    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)


def plot_section3(plot_errorbar, x_axis_time):
    data_name = collections.OrderedDict({"scitail_3":"SCITAIL"})

    x_axis_time = False

    data, avg_time = get_data(data_name)
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0

    f, axes = plt.subplots(1, 1, figsize=(10,18))

    one_plot(data["scitail_3"][10000],
             avg_time["scitail_3"][10000],
             10000,
             axes,
             "SciTail",
             0,
             reported_accuracy=[.65, .754, 0.705, 0.796],
             classifiers=['word overlap', 'dam', 'esim', 'DGEM'],
             plot_errorbar=False,
             legend_loc='upper left',
             xlim=[1, 100],
             relabel_scalar=True,
             fontsize=24,
             logx=True,
             x_axis_time=False)



    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)



def plot_section3_bidaf(plot_errorbar, x_axis_time):
    data_name = collections.OrderedDict({"bidaf_master_1":"BIDAF"})

    x_axis_time = False

    data, avg_time = get_data(data_name)
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0

    f, axes = plt.subplots(1, 1, figsize=(8,9))

    one_plot(data["bidaf_master_1"][10000],
             avg_time["bidaf_master_1"][10000],
             10000,
             axes,
             "SQuAD",
             0,
             reported_accuracy=[.677],
             classifiers=['glove'],
             plot_errorbar=False,
             legend_loc='lower right',
             xlim=[1, 100],
             logx=True,
             fontsize=24,
             relabel_scalar=True,
             x_axis_time=False)



    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)


def main(plot_errorbar, x_axis_time):
    x_axis_time = False
    data, avg_time = get_data()
    
    fig = plt.figure()

    data_sizes = list(data[list(data.keys())[0]].keys())
    data_sizes.sort()

    sqrt_num_plots = int(np.ceil(np.sqrt(len(data_sizes))))
    counter = 0
    for data_size in data_sizes:
        counter += 1

        cur_ax = fig.add_subplot(sqrt_num_plots,sqrt_num_plots,counter)
        cur_ax.set_xscale('log')

        experiment_counter = 0
        for cur_data in data_name:
            try:
                one_plot(data[cur_data][data_size],
                        avg_time[cur_data][data_size],
                        data_size,
                        cur_ax,
                        data_name[cur_data],
                        experiment_counter,
                        plot_errorbar=plot_errorbar,
                        x_axis_time=x_axis_time)
                experiment_counter += 1
            except:
                continue

    classifiers = get_classifiers(data[list(data.keys())[0]])
    save_plot(data_name, data[list(data.keys())[0]].keys(), classifiers, True, plot_errorbar=args.plot_errorbar, x_axis_time=args.x_axis_time)
        

def get_data(data_name):
    with_replacement = True

    all_data = collections.OrderedDict()
    all_avg_time = collections.OrderedDict()
    
    for cur_name in data_name:
        data, avg_time = samplemax.compute_sample_maxes(cur_name, with_replacement, return_avg_time=True)

        all_data[cur_name] = data
        all_avg_time[cur_name] = avg_time

    return all_data, all_avg_time

    
def one_plot(data, avg_time, data_size, cur_ax, data_name, experiment_counter, classifiers, logx, plot_errorbar, x_axis_time, legend_loc, relabel_scalar=False, reported_accuracy=None, encoder_name=None, fontsize=16, xlim=None):
    cur_ax.set_title(data_name, fontsize=fontsize)
    # classifiers = list(data.keys())
    max_first_point = 0
    classifier_counter = 0
    cur_ax.set_ylabel("Validation EM", fontsize=fontsize)
    
    if x_axis_time:
        cur_ax.set_xlabel("Seconds",fontsize=fontsize)
    else:
        cur_ax.set_xlabel("Hyperparameter assignments",fontsize=fontsize)
    if logx:
        cur_ax.set_xscale('log')
    for ix, classifier in enumerate(classifiers):
        cur_means = data[classifier]['mean']
        cur_vars = data[classifier]['var']
        
        if x_axis_time:
            times = [avg_time[classifier] * (i+1) for i in range(len(cur_means))]
        else:
            times = [i+1 for i in range(len(cur_means))]
        colors = ["#8c564b", '#1f77b4', '#ff7f0e', '#17becf']
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cur_color = colors[classifier_counter]
        classifier_counter += 1
        if classifier == 'linear':
            cur_classifier_name = 'logistic regression'
        elif classifier == 'dam':
            cur_classifier_name = "DAM"
        elif classifier == 'word overlap':
            cur_classifier_name = "n-gram baseline"
        elif classifier == 'esim':
            cur_classifier_name = "ESIM"
        elif classifier == 'glove':
            cur_classifier_name = "BIDAF"
        else:
            cur_classifier_name = classifier
        
        if reported_accuracy:
            cur_ax.plot([0, 10000], [reported_accuracy[ix], reported_accuracy[ix]], linestyle='--', linewidth=3, color=cur_color)
            plt.text(95,reported_accuracy[ix] + 0.003,f'reported {cur_classifier_name} accuracy', ha='right', style='italic', fontsize=fontsize-5, color=cur_color)

        if encoder_name:
            cur_classifier_name = encoder_name + " " + cur_classifier_name

        if plot_errorbar:
            # plt.fill_between(times, np.array(cur_means)-np.array(cur_vars), np.array(cur_means)+np.array(cur_vars), edgecolor='#1B2ACC', facecolor='#089FFF')

            line = cur_ax.errorbar(
                times, cur_means, yerr=cur_vars, label=cur_classifier_name,
                linestyle=linestyle[experiment_counter], color=cur_color)
        else:
            line = cur_ax.plot(times, cur_means,
                               label=cur_classifier_name, linestyle=linestyle[experiment_counter], linewidth=3, color=cur_color)
    # cur_ax.plot([0, 10000], [0.754, 0.754], label = 'reported dam accuracy', linestyle='--', color='red')
    # cur_ax.plot([0, 10000], [0.65, 0.65], linestyle='--', label = 'reported word overlap accuracy', color='green')

    
    left, right = cur_ax.get_xlim()
    if xlim:
        cur_ax.set_xlim(xlim)
        cur_ax.xaxis.set_ticks(np.arange(xlim[0], xlim[1]+5, 10))
    for tick in cur_ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in cur_ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    plt.locator_params(axis='y', nbins=5)
    from matplotlib.ticker import ScalarFormatter
    if relabel_scalar:
        
        for axis in [cur_ax.xaxis]:
            axis.set_ticks([1, 10, 50, 100])
            axis.set_major_formatter(ScalarFormatter())

    #bottom, top = cur_ax.get_ylim()
    #cur_ax.set_ylim(max_first_point, top)
    #cur_ax.set_ylim(bottom=0.725)

    #cur_ax.set_title("SST2")

    
    #cur_ax.legend(bbox_to_anchor=(1,0,.5,1))
    cur_ax.legend(loc=legend_loc, fontsize=fontsize)
    
    plt.tight_layout()

    
def save_plot(data_name, data_sizes, classifiers, with_replacement, x_axis_time, plot_errorbar):
    sizes = cat_list(data_sizes)
    cs = cat_list(classifiers)
    data_names = cat_list(data_name.keys())
    if x_axis_time:
        x_axis_units = "time"
    else:
        x_axis_units = "trials"

    if plot_errorbar:
        errorbar = "/errorbar"
    else:
        errorbar = ""
    save_loc = "plot_drafts/expected_max_dev{}/{}_{}_{}_x={}_maxx={}_replacement={}.pdf".format(
        errorbar, data_names, sizes, cs, x_axis_units, 1, with_replacement)
    print("saving to {}...".format(save_loc))
    plt.savefig(save_loc)


def cat_list(l):
    cat_l = ""
    l = list(l)
    l.sort()
    for cur_l in l:
        cat_l += str(cur_l).replace(" ","") + ","
    cat_l = cat_l[0:len(cat_l) - 1]
    return cat_l

def get_classifiers(data):
    classifiers = set()
    for data_size in data:
        for c in data[data_size].keys():
            classifiers.add(c)


    cs = list(classifiers)
    cs.sort()
    return classifiers


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_axis_time", action='store_true', default=True)
    parser.add_argument("--plot_errorbar", action='store_true', default=False)
    args = parser.parse_args()
    plot_section1(**vars(args))
    plot_section3(**vars(args))
    plot_section3_bcn(**vars(args))
    plot_section3_bidaf(**vars(args))