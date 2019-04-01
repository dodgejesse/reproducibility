import numpy as np
import pandas

def from_file(data_name = "sst_fiveway", return_lr_acc=False):
    if "imdb" in data_name:
        keys_and_indices = imdb_keys_and_indices()
    elif "ag_news" in data_name:
        keys_and_indices = ag_keys_and_indices()
    elif "battle" in data_name:
        keys_and_indices = battle_keys_and_indices()
    else: #"sst" in data_name or "hatespeech" in data_name:
        keys_and_indices = sst_keys_and_indices()

    key_names = get_key_names()
    filename = "/home/jessedd/data/reprocudibility/{}_search.{}".format(data_name, key_names["sep_name"])

    
    df = pandas.read_csv(filename, sep=key_names['sep'])
    data, avg_time, lr_acc = get_numtrain_to_classifier_to_field(df, key_names)

    
    if return_lr_acc:
        return data, lr_acc
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
            data[train_num][classifier] = df[key_names['dev_acc']][(df[key_names['classifier']] == classifier) & (df[key_names['train_num']] == train_num)]

            avg_time[train_num][classifier] = df[key_names['duration']][(df[key_names['classifier']] == classifier) & (df[key_names['train_num']] == train_num)].mean()
            
            lr = df[key_names['lr']][(df[key_names['classifier']] == classifier) & (df[key_names['train_num']] == train_num)]
            lr_acc[train_num][classifier] = []
            for i in range(len(lr)):
                lr_acc[train_num][classifier].append((lr.iloc[i], data[train_num][classifier].iloc[i]))
            
                
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
    data = from_file("hatespeech")
    import pdb; pdb.set_trace()
    print(data)

if __name__ == '__main__':
    main()
