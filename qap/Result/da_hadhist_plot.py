import numpy as np
from glob import glob

for lineID in range(5):
    maxStep = 40000
    # data = "had12"
    # lineID = 2
    betaStart_list = [0.001, 0.01, 0.1]
    betaStop_list  = [0.1 ,0.5, 1, 5, 10]
    # fileList = glob("da_hadhist{}*".format(maxStep))
    ARPD_array = np.zeros((len(betaStart_list),len(betaStop_list)))
    E_C_best_array = np.zeros((len(betaStart_list),len(betaStop_list)))

    for i, betaStart in enumerate(betaStart_list):
        for j, betaStop in enumerate(betaStop_list):
            filename = "da_hadhist{}betaStart{}betaStop{}.txt".format(maxStep,betaStart,betaStop)
            with open(filename, 'r') as f:
                lines = f.readlines()
                lines = [lines[lineID]]
                ARPD_array[i,j] = lines[0].split("\t")[-1]
                E_C_best_array[i,j] = lines[0].split("\t")[1]

    print(ARPD_array)
    ARPD_array_min = np.min(ARPD_array)
    ARPD_array_min_idx = np.argmin(ARPD_array)
    print("Best avg ARPD: {} @ idx: {}".format(ARPD_array_min, ARPD_array_min_idx))
    # print(E_C_best_array)
    # print(np.mean(E_C_best_array, 0))
    # print(np.mean(E_C_best_array, 1))

