import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from collections import defaultdict

gestures = ['stand', 'walkf', 'walkb', 'runf', 'runb', 'jump', 'kick', 'duck']

def print_speed_info(data):
    recordings, hits, speeds, extras = data
    speeds = np.array(speeds)
    hits = np.array(hits)
    speed_flat = speeds.flatten()
    speed_avg = np.average(speed_flat)
    speed_std = np.std(speed_flat)
    print('Speed avg: ' + str(speed_avg) + ' ' + str(speed_std))

    hits_flat = hits.flatten()
    hit_speed_flat = speed_flat * hits_flat
    hit_speed_avg = np.average(hit_speed_flat)
    hit_speed_std = np.std(hit_speed_flat)
    print('Hit speed avg: ' + str(hit_speed_avg) + ' ' + str(hit_speed_std))

def get_gesture_hits(data):
    recordings, hits, speeds, extras = data
    gesture_hits = defaultdict(list)
    for rs, hs in zip(recordings, hits):
        for r, h in zip(rs, hs):
            gesture_hits[r.label].append(h)
    for gesture in gesture_hits:
        gesture_hits[gesture] = np.array(gesture_hits[gesture])
    return gesture_hits

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def plot_gesture_hits(kb_gesture_hits, fc_gesture_hits):
    kb_avgs = np.array([np.average(kb_gesture_hits[g]) for g in gestures])
    kb_cnfs = np.array([confidence_interval(kb_gesture_hits[g]) for g in gestures])
    fc_avgs = np.array([np.average(fc_gesture_hits[g]) for g in gestures])
    fc_cnfs = np.array([confidence_interval(fc_gesture_hits[g]) for g in gestures])
    N = len(kb_avgs)

    fig, ax = plt.subplots()
    ind = np.arange(N)
    width = 0.35
    p1 = ax.bar(ind, kb_avgs, width, color='blue', bottom=0, yerr=kb_cnfs)
    p2 = ax.bar(ind+width, fc_avgs, width, color='orange', bottom=0, yerr=fc_cnfs)

    ax.set_title('Mock Game Hit Rate')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(gestures)
    ax.legend((p1[0], p2[0]), ('Keyboard', 'Finger People'))

    plt.show()

if __name__ == '__main__':
    kb_data = pickle.load(open('kb_summary.pickle', 'rb'))
    fc_data = pickle.load(open('fc_summary.pickle', 'rb'))

    print_speed_info(kb_data)
    print_speed_info(fc_data)

    kb_gesture_hits = get_gesture_hits(kb_data)
    fc_gesture_hits = get_gesture_hits(fc_data)

    plot_gesture_hits(kb_gesture_hits, fc_gesture_hits)
