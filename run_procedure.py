import numpy as np

from metrics import compute_delays_false_alarms, compute_delays_sequential

def performance_run(alg, dataset, threshold=np.inf):
    assert dataset.name == "gaussian" # TODO others
    data = dataset.get_data_cp()
    cps = dataset.get_cps()
    alg.threshold = np.inf
    S, _ = alg.compute_test_stat(data[0])

    fa, delay, nd = 0, 0, 0
    st = np.where(S > threshold)[0]
    if len(st) == 0:
        nd = 1
    elif st[0] < cps[0]:
        fa = 1
    delay = st[0] - cps[0]
    return  S, (st[0], fa, delay, nd, threshold)

def run_changepoint_occupancy(alg, dataset, threshold=np.inf):
    return_data = []
    min_diff = 10
    data = dataset.get_data_cp()
    cps = dataset.get_cps()
    if np.isinf(threshold):
        threshold = 0
    alg.threshold = threshold

    S_q = np.empty(0)
    change_points_q = []

    delays_q = np.empty(0)
    cur_cp_ind = 0
    false_alarms_q = 0

    st_q = 0
    new_st_q = 0
    detected_cnt = 0
    while new_st_q >= 0:

        X = data[st_q + 1:].copy()
        new_S_q, new_st_q = alg.compute_test_stat(X)
        print(new_S_q[-3:])
        S_q = np.append(S_q, new_S_q)
        
        if new_st_q > 0:
            
            detected_cnt += 1
            st_q += new_st_q
            print('Detected change point:', st_q)
            change_points_q += [int(st_q)]

            if (cur_cp_ind >= len(cps)\
                or (cps[cur_cp_ind] - st_q > min_diff)):
                print("False Alarm")
                false_alarms_q += 1
            else:
                if (cur_cp_ind < len(cps)\
                    and np.abs(cps[cur_cp_ind] - st_q) <= min_diff):
                    delays_q = np.append(delays_q, np.array([0.0]), axis=0)
                    cur_cp_ind += 1
                    continue

                delays_q = np.append(delays_q,\
                                    np.array([st_q - cps[cur_cp_ind]]),\
                                        axis=0)
                cur_cp_ind += 1

    nd = len(cps) - detected_cnt + false_alarms_q
    return_data.append(tuple([false_alarms_q, np.mean(delays_q), np.std(delays_q), nd, threshold]))
    return S_q, return_data

def run_changepoint(alg, dataset, threshold=np.inf):
    if dataset.name == "occupancy":
        return run_changepoint_occupancy(alg, dataset, threshold)
    data = dataset.get_data_cp()
    cps = dataset.get_cps()
    
    if dataset.type == "sequential":
        alg.threshold = threshold
        S_q = np.empty(0)
        detected_cps= []
        st_q = 0
        new_st_q = 0

        while new_st_q >= 0:
            X = data[st_q + 1:].copy()
            new_S_q, new_st_q = alg.compute_test_stat(X)
            print(len(data[st_q+1:]), len(new_S_q), new_st_q)
            S_q = np.append(S_q, new_S_q) 
            if new_st_q >= 0:
                st_q += new_st_q
                detected_cps += [int(st_q)]
        alg.threshold = np.inf
        return S_q, [(*compute_delays_sequential(cps, detected_cps), threshold)]
    
    elif dataset.type == "independent":
        S_q = []
        detected_cps = []
        for i in range(len(data)):
            S, _ = alg.compute_test_stat(data[i])
            S_q.append(S)
        return np.hstack(S_q), [(*compute_delays_false_alarms(len(data), S_q, threshold, cps), threshold)]

def tune_threshold(alg, data):
    smax = -np.inf
    for i in range(len(data)):
        S, _ = alg.compute_test_stat(data[i])
        smax = np.maximum(smax, np.max(S))
    threshold = smax
    return threshold

