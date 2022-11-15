import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import pickle
from glob import glob
from tqdm import tqdm
import sys
sys.path.insert(0, '..')
from DA4 import DA
from qap2qubo import *
from penalty_weight import *

def myDA(Q, init_bin, maxStep=1000, betaStart=0.1, betaStop=50, E_offset_increase_rate=0, plot_p=False):
    
    N = Q.shape[0]
    Q_norm_coef = la.norm(Q, 'fro')*N
    Q = Q/np.sqrt(Q_norm_coef)

    x = init_bin
    init_e = Q@init_bin
    
    betaList = np.linspace(betaStart, betaStop, maxStep)
    tempStart = 1/betaStart
    tempEnd = 1/betaStop
    decay = (tempEnd / tempStart)**(1/(maxStep-1))
    betaList = [1/(tempStart * decay**i) for i in range(maxStep)]

    E_offset = 0
    E_best = 999999999
    pList = []

    for idx_step, beta in enumerate(betaList):
        # I: Trial phase
        idx_ones = np.squeeze(x)
        sum_partial = np.sum(Q[:,idx_ones],1).reshape(-1,1)
        dE = np.multiply((2*(1-2*x)),sum_partial)+np.diag(Q).reshape(-1,1)
        # p = np.exp(-(dE-E_offset)*beta)
        # idx_rand = np.argmax(p)
        # print(idx_rand)
        p = np.minimum(np.exp(-(dE-E_offset)*beta), 1.) # Acceptance Probility 
        accepted = np.random.binomial(1, np.squeeze(p), N)
        # II: Update phase
        if( np.any(accepted) ):
        # if( ~idx_rand ):
            idx_rand = np.random.choice(accepted.nonzero()[0])
            x[idx_rand] ^= True
            E_offset = 0
            pList.append(p[idx_rand])
        else:
            E_offset += E_offset_increase_rate
            pList.append(0)

        E_current = x.T@Q@x
        if( E_current < E_best):
            E_best = E_current
            x_best = x 

    if plot_p:
        plt.figure(figsize=(8,4))
        plt.grid()
        plt.plot(pList, label="Acceptance Prob.", color='r', marker='.', linewidth=1.0, linestyle='')
        # plt.plot(1/np.array(betaList), label="Acceptance Prob.", color='r', marker='', linewidth=1.0, linestyle='-')
        plt.legend(loc='upper right')
        # plt.xticks(self.N*self.self.maxStep)
        plt.xlabel('MC steps', fontsize=13)
        plt.ylabel('Acceptance Prob.', fontsize=13)
        plt.show()

    return x_best

def bqp_test(path="../data/qapdata", file="had", iters=1000, spin=False, betaStart=0.01, betaStop=0.1, E_offset_increase_rate=0):

    output_file = open("Result/myda_{}hist{}betaStart{}betaStop{}E_offset_increase_rate{}.txt".format(file, iters, betaStart, betaStop, E_offset_increase_rate), "w")

    data_list = [tag for tag in glob("{}/{}*".format(path, file))]
    data_list = np.sort(data_list)
    
    run=1
    for data in tqdm(data_list):
        distribution = np.zeros(10)
        t = 0
        min_e = 9999999

        # ==== Load Flow, distance Matrix ====
        dataname = data.split("/")[-1].split(".")[0]
        F, D, n, E_opt = loadQAP(dataname)

        # ==== QUBO formulation ====
        C, C_offset, P, P_offset, N = qap2qubo(F, D, n)

        # ==== Penalty Weight ====
        w = penalty_weight(C, P, N, mode="moc")

        Q = C + w*P
        init_bin = np.ones((N,1), dtype=np.bool)

        E_C_list = []
        for n in tqdm(range(run)):
            #### annealing ####
            # da = DA(Q, init_bin, maxStep=iters,
                    #  betaStart=betaStart, betaStop=betaStop, kernel_dim=(32 * 2,), spin=spin, energy = init_e)
            # da.run()
            start_time = time.time()
            x = myDA(Q, init_bin, maxStep=iters, E_offset_increase_rate=E_offset_increase_rate)
            total_time = time.time() - start_time

            E_P = x.T@P@x + P_offset
            if( E_P==0 ):
                E_C = x.T@C@x
                E_C_list.append(E_C)

                print("solution", E_opt, E_C)
                if E_C < min_e :
                    min_e = E_C
                if (E_opt - E_C) >= 0:
                    l = 0
                    distribution[0] += 1
                    t += total_time
                    print('find optimal solution', E_opt, E_C)
                    break
                else:
                    l = int(np.ceil((abs((E_opt - E_C) / E_opt) * 100)))
                if l > 9:
                    l = 9
                distribution[l] += 1
                print(f"Time: {total_time}")
                t += total_time
            
        print("\n{}/{} feasible solutions.".format(len(E_C_list),run))
        # print("Avg. Energy:{} / Best Energy:{}".format(np.mean(E_C_list),np.min(E_C_list)))
        ARPD = np.mean((np.array(E_C_list)-E_opt)/E_opt)*100
        print("ARPD: {} %\n".format(ARPD))

        output_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(dataname, min_e, t, distribution, sum(distribution)/run, ARPD))
        output_file.flush()

if __name__ == '__main__':
    bqp_test(file="had12", iters=10000, E_offset_increase_rate=1)
    
    # E_offset_increase_rate_list = [0,0.01,0.1,1,10]
    # for E_offset_increase_rate in E_offset_increase_rate_list:
    #     bqp_test(file="had12", iters=10000, E_offset_increase_rate=E_offset_increase_rate)

    # betaStart_list = [0.001, 0.01, 0.1]
    # betaStop_list  = [0.1 ,0.5, 1, 5, 10]
    # for betaStart in betaStart_list:
    #     for betaStop in betaStop_list:
    #         bqp_test(file="had12", iters=40000, betaStart=betaStart, betaStop=betaStop, E_offset_increase_rate=0)