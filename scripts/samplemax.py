# this module will compute the distribution of the max, and also the expected value of the max, of a set of samples
# which were drawn IID from a distribution.

import load_data
import scipy.special

def compute_sample_maxes(data_name = "sst_fiveway", with_replacement = True):

    data = load_data.from_file(data_name)

    sample_maxes = {}
    for data_size in data:
        if data_size not in sample_maxes:
            sample_maxes[data_size] = {}
        for classifier in data[data_size]:

            sample_maxes[data_size][classifier] = sample_max(data[data_size][classifier], with_replacement)
    #import pdb; pdb.set_trace()
    return sample_maxes


# this implementation assumes sampling with replacement for computing the empirical cdf
def sample_max(cur_data, with_replacement):

    cur_data.sort()
    N = len(cur_data)
    pdfs = []
    for n in range(1,N+1):
        # the CDF of the max
        F_Y_of_y = []
        for i in range(1,N+1):

            if with_replacement:
                F_Y_of_y.append(cdf_with_replacement(i,n,N))
            else:
                F_Y_of_y.append(cdf_without_replacement(i,n,N))

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

def cdf_with_replacement(i,n,N):
    return (i/N)**n

def cdf_without_replacement(i,n,N):
    
    return scipy.special.comb(i,n) / scipy.special.comb(N,n)

if __name__ == '__main__':
    s_maxes = compute_sample_maxes()

    

    for data_size in s_maxes: 
        accuracies = {}
        for classifier in s_maxes[data_size]:
            print(classifier)
            for i in range(15):
                if i not in  accuracies:
                    accuracies[i] = []
                accuracies[i].append(s_maxes[data_size][classifier][i])
        for i in range(15): print(accuracies[i])

    import pdb; pdb.set_trace()
    
    print(s_maxes)
