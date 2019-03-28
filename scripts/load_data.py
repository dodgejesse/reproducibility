

def from_file(data_name = "sst_fiveway", return_lr_acc=False):
    if "imdb" in data_name:
        keys_and_indices = imdb_keys_and_indices()
    elif "ag_news" in data_name:
        keys_and_indices = ag_keys_and_indices()
    elif "battle" in data_name:
        keys_and_indices = battle_keys_and_indices()
    else: #"sst" in data_name or "hatespeech" in data_name:
        keys_and_indices = sst_keys_and_indices()

    filename = "/home/jessedd/data/reprocudibility/{}_search.{}".format(data_name, keys_and_indices["sep_name"])


        
    lines = []
    with open(filename, "r") as f:
        lines = f.readlines()
    data = process_lines(lines, keys_and_indices, return_lr_acc)
    return data

def process_lines(lines, keys_and_indices, return_lr_acc):
    print("lines with missing accuracy values:")
    possible_values = {}
    data = {}
    
    for line in lines:
        if "Task_Name" in line or "best_epoch" in line:
            first_line = line
            continue

        # to check that the columns are in the correct order, match the output of this against keys_and_indices
        check_column_nums = False
        if check_column_nums:
            split_first_line = first_line.split(keys_and_indices["sep"])
            split_line = line.split(keys_and_indices["sep"])
            for i in range(len(split_first_line)):
                print(i, split_first_line[i], "\t\t\t", split_line[i])
            import pdb; pdb.set_trace()
        
        update_possible_values(line, possible_values, keys_and_indices)
        extract_data(line, data, keys_and_indices)

    print("")

    debug = False
    if debug:
        print("num_experiments")
        print_num_experiments(data)
        
        #for k in possible_values:
        #    print(k, possible_values[k])
        #print("")
        import pdb; pdb.set_trace()
        #print("possible learning rates:")
        #for classifier in data[32]:
            #lrs = []
            #for example in data[32][classifier]:

    if return_lr_acc:
        return data, possible_values["lr_acc"]
    else:
        return data
        

def extract_data(line, data, keys_and_indices):
    #import pdb; pdb.set_trace()
    split_line = line.split(keys_and_indices["sep"])
    try:
        cur_data_size = int(split_line[keys_and_indices["data_size"]])
    except:
        cur_data_size = split_line[keys_and_indices["data_size"]]
    cur_classifier = split_line[keys_and_indices["classifier"]]

    if split_line[keys_and_indices["accuracy"]] == '':
        print(cur_data_size, cur_classifier)
        return

    else:
        cur_accuracy = float(split_line[keys_and_indices["accuracy"]])

    if cur_data_size not in data:
        data[cur_data_size] = {}

    if cur_classifier not in data[cur_data_size]:
        data[cur_data_size][cur_classifier] = []

    data[cur_data_size][cur_classifier].append(cur_accuracy)
        

def update_possible_values(line, possible_values, keys_and_indices):
    split_line = line.split(keys_and_indices["sep"])
    for key_name in ["experiment_name", "experiment_id", "classifier", "data_size", "lr"]:
        if key_name not in possible_values:
            possible_values[key_name] = []
        possible_values[key_name].append(split_line[keys_and_indices[key_name]])
    if "lr_acc" not in possible_values:
        possible_values["lr_acc"] = {}
    cur_class = split_line[keys_and_indices["classifier"]]
    if cur_class not in possible_values["lr_acc"]:
        possible_values["lr_acc"][cur_class] = []
    possible_values["lr_acc"][cur_class].append((float(split_line[keys_and_indices["lr"]]), float(split_line[keys_and_indices["accuracy"]])))
    


def print_num_experiments(data):
    counters = {}
    for data_size in data:
        for classifier in data[data_size]:
            counters[str(data_size) + "_" + classifier] = len(data[data_size][classifier])
    
    #import pdb; pdb.set_trace()
    for counter in counters:
        print(counter, counters[counter])

def imdb_keys_and_indices():
    return {
        "experiment_name": 2,
        "experiment_id": 3,
        "classifier":-11,
        "data_size": -2,
        "accuracy": 4,
        "sep":",",
        "sep_name":"csv"
    }

def ag_keys_and_indices():
    return {
        "experiment_name": 3,
        "experiment_id": 4,
        "classifier":-11,
        "data_size": -2,
        "accuracy": 5,
        "sep":",",
        "sep_name":"csv"
    }

def sst_keys_and_indices():
    return {
        "experiment_name": 0,
        "experiment_id": 0,
        "classifier":25,
        "data_size": 5,
        "accuracy": 2,
        "lr":45,
        "sep":"\t",
        "sep_name":"tsv"
    }

def battle_keys_and_indices():
    return {
        "experiment_name": 0,
        "experiment_id": 0,
        "classifier":24,
        "data_size": 5,
        "accuracy": 2,
        "lr":50,
        "sep":"\t",
        "sep_name":"tsv"
    }


def main():
    #data = from_file("hatespeech_10k")
    data = from_file("battle_year")
    import pdb; pdb.set_trace()
    print(data)

if __name__ == '__main__':
    main()
