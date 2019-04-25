# this module will compute the distribution of the max, and also the expected value of the max, of a set of samples
# which were drawn IID from a distribution.

import load_data
import scipy.special
import numpy as np

# used to check the bootstrap approximates the mean and var we compute exactly.
def compute_bootstrap(data_name):
    data = load_data.from_file(data_name, return_avg_time = True)

    classifier_to_performance = data[0][6919]

    # just for comparison:
    closed_form_mean_var, avg_times = compute_sample_maxes(data_name, True, True)

    c_to_means = {}
    c_to_vars = {}
    for num_samples in [50000]:
        for classifier in classifier_to_performance:
            bootstrap_means = []
            bootstrap_vars = []
            for n in range(len(classifier_to_performance[classifier])):
                cur_mean, cur_std = draw_bootstrap_samples(classifier_to_performance[classifier], n+1, num_samples)
                bootstrap_means.append(cur_mean)
                bootstrap_vars.append(cur_std)


            c_to_means[classifier] = bootstrap_means
            c_to_vars[classifier] = bootstrap_vars

            #print(closed_form[0][6919][classifier])
            #print(bootstrap_vals)
            #diffs = [closed_form[0][6919][classifier][i] - bootstrap_means[i] for i in range(len(bootstrap_means))]
            #print(sum(abs(np.asarray(diffs))))
                        
            #print(diffs)
            
    import pdb; pdb.set_trace()
    

def draw_bootstrap_samples(cur_data, n, num_samples):

    samples = np.random.choice(cur_data, (num_samples, n))
    maxes = np.max(samples, 1)
    return np.mean(maxes), np.std(maxes)

    

def compute_sample_maxes(data_name = "sst2", with_replacement = True, return_avg_time = False):

    data = load_data.from_file(data_name, return_avg_time = return_avg_time)
    if return_avg_time:
        avg_time = data[1]
        data = data[0]

    sample_maxes = {}
    for data_size in data:
        if data_size not in sample_maxes:
            sample_maxes[data_size] = {}
        for classifier in data[data_size]:
            sample_maxes[data_size][classifier] = sample_max(data[data_size][classifier], with_replacement)
    #import pdb; pdb.set_trace()
    if return_avg_time:
        return sample_maxes, avg_time
    else:
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


    var_of_max_cond_n = compute_variance(N, cur_data, expected_max_cond_n, pdfs)

        
    return {"mean":expected_max_cond_n, "var":var_of_max_cond_n}

def cdf_with_replacement(i,n,N):
    return (i/N)**n

def cdf_without_replacement(i,n,N):
    
    return scipy.special.comb(i,n) / scipy.special.comb(N,n)

# this computes the standard error of the max.
# this is what the std dev of the bootstrap estimates of the mean of the max converges to, as
# is stated in the last sentence of the summary on page 10 of http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf
# uses equation 
def compute_variance(N, cur_data, expected_max_cond_n, pdfs):
    variance_of_max_cond_n = []
    for n in range(N):
        # for a given n, estimate variance with \sum(p(x) * (x-mu)^2), where mu is \sum(p(x) * x).
        cur_var = 0
        for i in range(N):
            cur_var += (cur_data[i] - expected_max_cond_n[n])**2 * pdfs[n][i]
        cur_var = np.sqrt(cur_var)

        variance_of_max_cond_n.append(cur_var)

    return variance_of_max_cond_n
    

if __name__ == '__main__':
    #bootstrap_results = compute_bootstrap("sst2_biattentive_elmo_transformer")
    

    
    s_maxes = compute_sample_maxes("sst2", True, True)
    import pdb; pdb.set_trace()

    

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
