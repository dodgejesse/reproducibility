import numpy as np
import pandas

def from_file(data_name = "hatespeech", return_lr_acc=False):

    key_names = get_key_names()
    filename = "/home/jessedd/data/reprocudibility/{}_search.{}".format(data_name, key_names["sep_name"])

    
    df = pandas.read_csv(filename, sep=key_names['sep'])
    data, avg_time, lr_acc = get_numtrain_to_classifier_to_field(df, key_names)

    
    if return_lr_acc:
        return data, lr_acc
    elif return_avg_time:
        return data, avg_time
    else:
        return data

def get_numtrain_to_classifier_to_field(df, key_names):
    data = {}
    avg_time = {}
    lr_acc = {}
    for train_num in df[key_names['train_num']].unique():
        if train_num not in data:
            data[train_num] = {}
            avg_time[train_num] = {}
            lr_acc[train_num] = {}
        for classifier in df[key_names['classifier']].unique():
            # the locations of the current experiments
            cur_locs = (df[key_names['classifier']] == classifier) & (df[key_names['train_num']] == train_num)
            
            data[train_num][classifier] = df[key_names['dev_acc']][cur_locs].values.tolist()

            avg_time[train_num][classifier] = df[key_names['duration']][cur_locs].mean()
            
            lr = df[key_names['lr']][cur_locs]
            lr_acc[train_num][classifier] = []
            for i in range(len(lr)):
                lr_acc[train_num][classifier].append((lr.iloc[i], data[train_num][classifier][i]))

                
    return data, avg_time, lr_acc


def get_key_names():
    return {'duration':'training_duration',
            'dev_acc':'best_validation_accuracy',
            'classifier':'model.encoder.architecture.type',
            'train_num':'dataset_reader.sample',
            'lr':'trainer.optimizer.lr',
            'sep':'\t',
            'sep_name':'tsv'
    }

def main():
    data = from_file("sst5")
    import pdb; pdb.set_trace()
    print(data)

if __name__ == '__main__':
    main()
