

def from_file(filename = "/home/jessedd/data/reprocudibility/imdb_search.csv"):
    if "imdb" in filename:
        keys_and_indices = imdb_keys_and_indices()
    lines = []
    with open(filename, "r") as f:
        lines = f.readlines()
    data = process_lines(lines, keys_and_indices)
    return data

def process_lines(lines, keys_and_indices):
    possible_values = {}
    data = {}
    for line in lines:
        if line.startswith("Task_Name"):
            continue
        update_possible_values(line, possible_values, keys_and_indices)
        extract_data(line, data, keys_and_indices)

    if False:
        print_num_experiments(data)
        
        for k in possible_values:
            print(k, possible_values[k])
            
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
            counters[data_size + "_" + classifier] = len(data[data_size][classifier])

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
    

def main():
    data = from_file()

if __name__ == '__main__':
    main()
