# this module will compute the distribution of the max, and also the expected value of the max, of a set of samples
# which were drawn IID from a distribution.

import load_data

def compute_sample_maxes():
    data = load_data.from_file()

    sample_maxes = {}
    for data_size in data:
        if data_size not in sample_maxes:
            sample_maxes[data_size] = {}
        for classifier in data[data_size]:

            sample_maxes[data_size][classifier] = sample_max(data[data_size][classifier])
    return sample_maxes
            
def sample_max(cur_data):

    cur_data.sort()
    N = len(cur_data)
    pdfs = []
    for n in range(1,N+1):
        # the CDF of the max
        F_Y_of_y = []
        for i in range(1,N+1):
            F_Y_of_y.append((i/N)**n)

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
            cur_expected += cur_data[i] * pdfs[n][i]
        expected_max_cond_n.append(cur_expected)
    return expected_max_cond_n

if __name__ == '__main__':
    compute_sample_maxes()
    
