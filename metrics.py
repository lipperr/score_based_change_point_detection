import numpy as np
from collections.abc import Iterable


def compute_delays_false_alarms(runs, S_list, threshold, cps):
    delays = []
    false_alarms = 0
    not_detected = 0

    for i in range(runs):

        S = S_list[i]
        res = np.ma.flatnotmasked_edges(np.ma.masked_array(S, S <= threshold))
        if isinstance(res, Iterable):
            imin, _ = res
            if imin - cps[i] < 0:
                print(imin, i)
                false_alarms += 1
            else:
                delays.append(imin - cps[i])
        else:
            not_detected +=1

    if not_detected == runs:
        return 0, 0, 0, not_detected

    if len(delays) == 0:
        delays.append(-1)

    delays = np.array(delays)
    mean_delay = np.round(delays.mean(), 1)
    std_delay = np.round(delays.std(), 1)

    return false_alarms, mean_delay, std_delay, not_detected

def compute_delays_sequential(true_cp, detected_cp):
    false_alarms = 0
    not_detected = 0
    delays = []
    cp_all = np.append(true_cp, detected_cp)
    pattern = np.append(np.zeros(len(true_cp)), np.ones(len(detected_cp)))

    ind_sorted = np.argsort(cp_all)
    cp_sorted = cp_all[ind_sorted]
    pattern_sorted = pattern[ind_sorted]
    if pattern_sorted[0] == 1:
        false_alarms += 1
        
    for i in range(1, len(pattern_sorted)):

        # Correctly detected change point
        if (pattern_sorted[i] == 1) and (pattern_sorted[i - 1] == 0):
            delays += [cp_sorted[i] - cp_sorted[i - 1]]
            
        # False alarm
        if (pattern_sorted[i] == 1) and (pattern_sorted[i - 1] == 1):
            # print("False Alarm", cp_sorted[i])
            false_alarms += 1
            
        # Non-detected change point
        if (pattern_sorted[i] == 0) and (pattern_sorted[i - 1] == 0):
            delays += [cp_sorted[i] - cp_sorted[i - 1]]
            not_detected += 1
            # print("Not detected", cp_sorted[i-1])
    
    # Last change point not detected
    if (pattern_sorted[-1] == 0):
        not_detected += 1
        # print("Not detected", cp_sorted[i-1])

    if len(delays) == 0:
        delays.append(-1)
    delays = np.array(delays)
    delays_mean = np.mean(delays)
    delays_std = np.std(delays)
    
    return false_alarms, delays_mean, delays_std, not_detected


