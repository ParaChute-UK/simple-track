import datetime as dt

import numpy as np


def load_mwe(filename):
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    data = np.loadtxt(filename)
    mwe_idx = str(filename)[-7]
    time = base_time + dt.timedelta(minutes=5 * (int(mwe_idx) - 1))
    return time, data


def load_all_mwe(filenames):
    base_time = dt.datetime(2024, 1, 1, 0, 0, 0)
    all_data = [np.loadtxt(fnm) for fnm in filenames]
    all_data = np.array(all_data)

    all_times = [
        base_time + dt.timedelta(minutes=5 * int(mwe_idx))
        for mwe_idx in range(len(filenames))
    ]
    return all_times, all_data
