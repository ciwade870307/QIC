import time
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm
import sys
sys.path.insert(0, '..')
from DA4 import DA
from qap2qubo import *
from penalty_weight import *
from qubo_gen import *
from SQA import *

def bqp_test(path="../data/qapdata", file="had", iters=1000, spin=False, M=4, T=0.15):

    output_file = open("Result/sqa_{}hist{}M{}T{}.txt".format(file, iters, M, T), "w")

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

        # DA setup
        # Q = np.zeros((N+1,N+1))
        # Q[0:N,0:N] = C + w*P
        # init_bin = np.zeros((N+1)).astype(np.float32)
        # init_e = init_bin @ Q
        # SQA setup
        Q = C + w*P
        J, h, offset = qubo2ising(Q)
        steps = iters
        Gamma0 = 10
        Gamma1 = 1e-8
        decay_rate = (Gamma1 / Gamma0)**(1/(steps-1))
        schedule = [Gamma0 * decay_rate**i for i in range(steps)]
        F = 1
        G = False

        E_C_list = []
        for n in tqdm(range(run)):
            #### annealing ####
            start_time = time.time()
            spin = one_SQA_run(J, h, schedule, M, T, field_cycling=F, return_pauli_z=True)
            total_time = time.time() - start_time
            x = np.array(spin)>=1
            print(f"x:{x}")
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
                t += total_time
            
        print("\n{}/{} feasible solutions.".format(len(E_C_list),run))
        print("Avg. Energy:{} / Best Energy: X".format(np.mean(E_C_list)))
        # print("Avg. Energy:{} / Best Energy:{}".format(np.mean(E_C_list),np.min(E_C_list)))
        ARPD = np.mean((np.array(E_C_list)-E_opt)/E_opt)*100
        print("ARPD: {} %\n".format(ARPD))

        output_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(dataname, min_e, t, distribution, sum(distribution)/run, ARPD))
        output_file.flush()

if __name__ == '__main__':
    bqp_test(file="had12", iters=10000)
