

def from_file(data_name = "imdb"):
    if "imdb" in data_name:
        keys_and_indices = imdb_keys_and_indices()
    elif "ag_news" in data_name:
        keys_and_indices = ag_keys_and_indices()
    else:
        assert False, "data_name should be one of imbd, ag_news"
    filename = "/home/jessedd/data/reprocudibility/{}_search.csv".format(data_name)


        
    lines = []
    with open(filename, "r") as f:
        lines = f.readlines()
    data = process_lines(lines, keys_and_indices)
    return data

def process_lines(lines, keys_and_indices):
    print("lines with missing accuracy values:")
    possible_values = {}
    data = {}
    for line in lines:
        if "Task_Name" in line:
            continue
        update_possible_values(line, possible_values, keys_and_indices)
        extract_data(line, data, keys_and_indices)

    print("")
        
    if True:
        print("num_experiments")
        print_num_experiments(data)
        
        for k in possible_values:
            print(k, possible_values[k])
        print("")

    return data
        

def extract_data(line, data, keys_and_indices):
    split_line = line.split(",")
    cur_data_size = int(split_line[keys_and_indices["data_size"]])
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
    split_line = line.split(",")
    for key_name in keys_and_indices:
        if key_name == "accuracy":
            continue
        if key_name not in possible_values:
            possible_values[key_name] = set()
        possible_values[key_name].add(split_line[keys_and_indices[key_name]])


def print_num_experiments(data):
    counters = {}
    for data_size in data:
        for classifier in data[data_size]:
            counters[str(data_size) + "_" + classifier] = len(data[data_size][classifier])

    for counter in counters:
        print(counter, counters[counter])

def imdb_keys_and_indices():
    return {
        "experiment_name": 2,
        "experiment_id": 3,
        "classifier":-11,
        "data_size": -2,
        "accuracy": 4,
    }

def ag_keys_and_indices():
    return {
        "experiment_name": 3,
        "experiment_id": 4,
        "classifier":-11,
        "data_size": -2,
        "accuracy": 5,
    }


def main():
    data = from_file()

if __name__ == '__main__':
    main()
