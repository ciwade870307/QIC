import numpy as np

def penalty_weight(C, P, N, mode="moc"):
    # Refer to "Penalty Weights in QUBO formulations: Permutation Problems"

    # Symmetric matrix to upper triangle matirx 
    C_triu = 2*np.triu(C, 1) + np.diag(np.diag(C))
    P_triu = 2*np.triu(P, 1) + np.diag(np.diag(P))

    # UB
    w_ub = np.sum(abs(C_triu))
    # MQC (for TSP)
    w_mqc = np.max(C_triu)
    # VLM
    Wc = []
    for i in range(N):
        # min
        idx_min = C_triu[i,:] < 0
        idx_min[i] = 0
        Wc.append( -C[i,i]-np.sum(C_triu[i,idx_min]) )
        # Max
        idx_max = C_triu[i,:] > 0
        idx_max[i] = 0
        Wc.append( C[i,i]+np.sum(C_triu[i,idx_max]) )
    Wc_np = np.array(Wc)
    w_vlm = np.max(Wc_np)
    # MOMC
    Wp = []
    for i in range(N):
        # min
        idx_min = P_triu[i,:] < 0
        idx_min[i] = 0
        Wp.append( -P[i,i]-np.sum(P_triu[i,idx_min]) ) 
        # Max
        idx_max = P_triu[i,:] > 0
        idx_max[i] = 0
        Wp.append( P[i,i]+np.sum(P_triu[i,idx_max]) )
    Wp_np = np.array(Wp)
    gamma = np.min(Wp_np[Wp_np>0])
    w_momc = max(1, w_vlm)/float(gamma)
    # MOC
    W_abs_ratio = abs(Wc_np[Wp_np>0]/Wp_np[Wp_np>0])
    w_moc = max(1, np.max(W_abs_ratio))


    if(mode == "ub"):
        w = w_ub
    elif(mode == "mqc"):
        w = w_mqc
    elif(mode == "vlm"):
        w = w_vlm
    elif(mode == "momc"):
        w = w_momc
    elif(mode == "moc"):
        w = w_moc 
    else:
        raise ValueError("No such penaly weight policy")

    return w