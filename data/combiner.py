import numpy as np
import pandas as pd
import glob
import random
import time
import matplotlib.pyplot as plt
import pickle5 as pickle

def split(foldername):
    fnames = glob.glob("{}2022*.pkl".format(foldername))
    for i in range(len(fnames)):
        train_data_all = combine_dataset([fnames[i]])
        test_data_all = combine_dataset(fnames[:i] + fnames[i+1:])
        print(fnames[i],fnames[:i] + fnames[i+1:])

def combine_dataset(fnames):
    data_all = []
    for i, filename in enumerate(fnames):
        # data = pd.read_pickle(filename)
        data = pickle.load(open(filename, "rb"))
        s = data['s_raw']
        mag = data['s11']
        phase = data['s11_phase']
        data['calibration_s_raw'] = [s[0]]*len(s)
        data['calibration_s11'] = [mag[0]]*len(s)
        data['calibration_s11_phase'] = [phase[0]]*len(s)
        data = data[10:]
        data_all.append(data)

    data_all = pd.concat(data_all)
    data_all.index = range(len(data_all.index))
    return data_all


if __name__ == "__main__":
    # foldername = "./user_study_setup/trackpad/narrower/"
    foldername = "./data/user_study/p2/hand_pose/"
    fnames = glob.glob("{}2022*.pkl".format(foldername))
    # split(foldername)

    train_data_all = combine_dataset(fnames)
    test_data_all = combine_dataset(fnames[:2])

    train_data_all.to_pickle(foldername+"train.pkl")
    test_data_all.to_pickle(foldername+"test.pkl")