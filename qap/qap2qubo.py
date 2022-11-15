import numpy as np

def loadQAP(dataname):
    datapath = f"../data/qapdata/{dataname}.dat"
    solnpath = f"../data/qapsoln/{dataname}.sln"

    print("========== Start loading: '{}' ==========".format(dataname))
    # Load Data
    dataList = []
    for line in open(datapath, 'r'):
        dataList += line.split()
    n = int(dataList[0])                                 # Dimension
    F = np.array(dataList[1:n**2+1], dtype="double").reshape(n,n)   # Flow Matrix
    D = np.array(dataList[n**2+1:], dtype="double").reshape(n,n)   # Dist Matrix

    # Load Optimal Sol
    solnSet = [line.split() for line in open(solnpath).readlines()]
    E_opt = float(solnSet[0][1])
    # opt_tmp = np.array(solnSet[1],dtype="int")
    # X_opt = np.zeros((n,n))
    # for i in range(n):
        # X_opt[i,opt_tmp[i]-1] = 1
    print(f"E_opt: {E_opt}")
    # print(f"X_opt:\n {X_opt}")

    return F, D, n, E_opt

def qap2qubo(F, D, n):

    N = n*n # Num of Qbits

    # Cost Matrix
    C = np.kron(F,D)
    C_offset = 0

    # Penalty Matrix (Constraint)
    subP = np.ones((n,n)) - 2*np.eye(n)
    P1= np.kron(np.eye(n),subP)
    P2= np.kron(subP, np.eye(n))
    P = P1+P2
    P_offset = 2*n

    return C, C_offset, P, P_offset, N

##########################################################################################################
if __name__ == "__main__":
    dataname = "lipa40a" #"esc16b", "nug12"
    F, D, n, E_opt = loadQAP(dataname)
    C, C_offset, P, P_offset, N = qap2qubo(F, D, n)