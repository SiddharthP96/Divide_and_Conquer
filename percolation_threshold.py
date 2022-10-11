import numpy as np
import networkx as nx
import random 
import sys

def ordinary_bond_percolation (G, p):
    H = nx.Graph()
    for n in G:
        H.add_node(n)
    for e in G.edges():
        if random.random() <= p:
            H.add_edge(e[0], e[1])
    return H

def cluster_sizes (H):
    P_infty = 0.0
    S = 0.0
    if len(H) > 0:
        components = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        if components[0] > 0:
            P_infty = float(components[0])
        if len(components)>1:
            if components[1] > 0:
                S = float(components[1])
    return P_infty, S

def multiple_realizations_percolation_model (G, p, T, model):
    if model != 'site' and model != 'bond':
        print ('Please specify the percolation model')
        sys.exit(0)
    av = 0.0
    susc = 0.0
    sec = 0.0
    N = float(len(G))
    t = 0
    while t < T:
        if model == 'bond':
            H = ordinary_bond_percolation (G, p)
        
        P_infty, S = cluster_sizes (H)
        
        av += P_infty / N
        susc += (P_infty / N)*(P_infty / N)
        sec += S / N
        t += 1
    av /= T
    susc /= T
    sec /= T
    susc -= av * av
    if av >0.0:
        susc /= av
    return av, susc, sec  

def newman_ziff_merge_clusters (clusters, cluster_id, n, m):
    
    if cluster_id[n] == cluster_id[m]:
        return cluster_id[n], len(clusters[cluster_id[n]])
    
    s = cluster_id[n]
    q = cluster_id[m]
    if len(clusters[s]) < len(clusters[q]):
        s = cluster_id[m]
        q = cluster_id[n]
        
    for t in clusters[s]:
        clusters[q][t] = 1
        cluster_id[t] = q
    clusters[s].clear()
    
    return q, len(clusters[q])


def newman_ziff_bond_percolation (G, res_av, res_susc):
    N = float(len(G))
    list_of_edges = []
    for e in G.edges():
        list_of_edges.append([e[0], e[1]])
    random.shuffle(list_of_edges)
    clusters = {}
    cluster_id = {}
    for n in G:
        cluster_id[n] = n
        clusters[n] = {}
        clusters[n][n] = 1
    P_infty = 1
    for i in range(0, len(list_of_edges)):
            n = list_of_edges[i][0]
            m = list_of_edges[i][1]
            q, P = newman_ziff_merge_clusters (clusters, cluster_id, n, m)
            if P > P_infty:
                P_infty = P
            res_av[i] += P_infty / N
            res_susc[i] += (P_infty / N) * (P_infty / N)

def newman_ziff_site_percolation (G, res_av, res_susc):
    
    N = float(len(G))
    
    list_of_nodes = list(G.nodes())
    random.shuffle(list_of_nodes)
    present = {}
    for n in list_of_nodes:
        present[n] = 0
    
    clusters = {}
    cluster_id = {}        
    P_infty = 1
    
    for i in range(0, len(list_of_nodes)):
        n = list_of_nodes[i]
        present[n] = 1
        cluster_id[n] = n
        clusters[n] = {}
        clusters[n][n] = 1
        for m in G.neighbors(n):
            if present[m] > 0:
                q, P = newman_ziff_merge_clusters (clusters, cluster_id, n, m)
                if P > P_infty:
                    P_infty = P
        res_av[i] += P_infty / N
        res_susc[i] += (P_infty / N) * (P_infty / N)
            


def multiple_newman_ziff_percolation (G, T, model):
    if model == 'bond':
        res_av = np.zeros(G.number_of_edges())
        res_susc = np.zeros(G.number_of_edges())
        for t in range(0, T):
            newman_ziff_bond_percolation (G, res_av, res_susc)
        for i in range(0, len(res_av)):
            res_av[i] /= float(T)
            res_susc[i] /= float(T)
            res_susc[i] -= res_av[i] * res_av[i]
            if res_av[i] >0.0:
                res_susc[i] /= res_av[i]
    
    elif model== 'site':
        res_av = np.zeros(G.number_of_nodes())
        res_susc = np.zeros(G.number_of_nodes())
        for t in range(0, T):
            newman_ziff_site_percolation (G, res_av, res_susc)
        for i in range(0, len(res_av)):
            res_av[i] /= float(T)
            res_susc[i] /= float(T)
            res_susc[i] -= res_av[i] * res_av[i]
            if res_av[i] >0.0:
                res_susc[i] /= res_av[i]
                
    return res_av, res_susc     



def test_merger (clusters, cluster_id, n, m):
    return len(clusters[cluster_id[n]]) * len(clusters[cluster_id[m]])
   


def explosive_percolation (G, res_av, res_susc):
    
    N = float(len(G))
    
    list_of_edges = []
    for e in G.edges():
        list_of_edges.append([e[0], e[1]])
    
    clusters = {}
    cluster_id = {}
    for n in G:
        cluster_id[n] = n
        clusters[n] = {}
        clusters[n][n] = 1
        
    P_infty = 1
    i = 0
    while len(list_of_edges) >= 2:
        
        e1 = random.randint(0, len(list_of_edges)-1)
        e2 = e1
        while e2 == e1:
            e2 = random.randint(0, len(list_of_edges)-1)
        
        ##test 1
        n = list_of_edges[e1][0]
        m = list_of_edges[e1][1]
        s1 = test_merger (clusters, cluster_id, n, m)
        
        ##test 1
        n = list_of_edges[e2][0]
        m = list_of_edges[e2][1]
        s2 = test_merger (clusters, cluster_id, n, m)
        
        
        if s1 < s2:
            n = list_of_edges[e1][0]
            m = list_of_edges[e1][1]
            list_of_edges[e1][0] = list_of_edges[len(list_of_edges)-1][0]
            list_of_edges[e1][1] = list_of_edges[len(list_of_edges)-1][1]
        else:
            n = list_of_edges[e2][0]
            m = list_of_edges[e2][1]
            list_of_edges[e2][0] = list_of_edges[len(list_of_edges)-1][0]
            list_of_edges[e2][1] = list_of_edges[len(list_of_edges)-1][1]
            
        del list_of_edges[-1]
        
    
            
   
        q, P = newman_ziff_merge_clusters (clusters, cluster_id, n, m)
        i = i + 1
    
        if P > P_infty:
            P_infty = P
        res_av[i] += P_infty / N
        res_susc[i] += (P_infty / N) * (P_infty / N)
            
            
def multiple_explosive_percolation (G, T):
    
    
    res_av = np.zeros(G.number_of_edges())
    res_susc = np.zeros(G.number_of_edges())
    for t in range(0, T):
        explosive_percolation (G, res_av, res_susc)
    for i in range(0, len(res_av)):
        res_av[i] /= float(T)
        res_susc[i] /= float(T)
        res_susc[i] -= res_av[i] * res_av[i]
        if res_av[i] >0.0:
            res_susc[i] /= res_av[i]
                
    
                
                
    return res_av, res_susc     










def percolate(g,iters=100,type='bond'):
    if type=='bond':
        res_av_bond, res_susc_bond = multiple_newman_ziff_percolation (g, iters, 'bond')
        return np.argmax(res_susc_bond)/len(res_susc_bond)

    elif type=='site':
        res_av_bond, res_susc_bond = multiple_newman_ziff_percolation (g, iters, 'site')
        return np.argmax(res_susc_bond)/len(res_susc_bond)

    elif type=='explosive':
        res_av_bond, res_susc_bond = multiple_explosive_percolation (g, iters)
        return np.argmax(res_susc_bond)/len(res_susc_bond)

