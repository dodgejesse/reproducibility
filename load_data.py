

def from_file(filename = "/home/jessedd/data/reprocudibility/imdb_search.csv"):
    if "imdb" in filename:
        keys_and_indices = imdb_keys_and_indices()
    lines = []
    with open(filename, "r") as f:
        lines = f.readlines()
    process_lines(lines, keys_and_indices)
    print(len(lines))

def process_lines(lines, keys_and_indices):
    possible_values = {}
    data = {}
    for line in lines:
        # the lines 
        #if '"' in line:
        update_possible_values(line, possible_values, keys_and_indices)
        extract_data(line, data, keys_and_indices)
    for k in possible_values:
        print(k, possible_values[k])
        

def extract_data(line, data, keys_and_indices):
    split_line = line.split(",")
    if "data_size" not in data:
        data["data_size"] = {}

    cur_data_size = split_line[keys_and_indices["data_size"]]
    if cur_data_size not in data["data_size"]:
        data["data_size"][cur_data_size] = {}

    if split_line[keys_and_indices["data_size"]] not in data["data_size"]:
        

def update_possible_values(line, possible_values, keys_and_indices):
    split_line = line.split(",")
    for key_name in keys_and_indices:
        if key_name not in possible_values:
            possible_values[key_name] = set()
        possible_values[key_name].add(split_line[keys_and_indices[key_name]])

    # note this method of splitting doesn't work for the aggregations.
    
    #possible_values["experiment_name"].add(split_line[2])
    #possible_values["experiment_id"].add(split_line[3])
    #possible_values["throttle"].add(split_line[-2])
    

def imdb_keys_and_indices():
    return {
        "experiment_name": 2,
        "experiment_id": 3,
        "classifier":-11,
        "data_size": -2
    }
    

def main():
    data = from_file()

if __name__ == '__main__':
    main()
