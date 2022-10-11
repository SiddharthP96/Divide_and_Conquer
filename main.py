import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import metis
from igraph import *
import powerlaw
import os 
import sys
import scipy.stats as stats
import pandas as pd
import copy
import pickle5 as pickle
from multiprocessing import Pool
import networkx.algorithms.community as nx_comm
import community as community_louvain
import mercator
from percolation_threshold import *
from methods import *

#ICM Spreading model
###########################################################################
def IC(g,S,p,mc=100):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """
    
    # Loop over the Monte-Carlo Simulations
    spread = []
    for it in range(mc):
        # Simulate propagation process      
        new_active, A = copy.deepcopy(S), copy.deepcopy(S)
        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:
                # Determine neighbors that become infected
                #np.random.seed(it)
                
                success = np.random.uniform(0,1,len(g.neighbors(node))) < p
                new_ones += list(np.extract(success, g.neighbors(node)))

            new_active = list(set(new_ones) - set(A))
            
            # Add newly activated nodes to the set of activated nodes
            A += new_active
            
        spread.append(len(A))
        
    return(np.mean(spread))



#Algorithms to detect influential spreaders given a hyperbolic embbedding 
#############################################################################
#x: dictionary with hyperbolic embeddings {node: (radial_i,angular_i,)}
#g: iGraph graph object
#k: number of spreaders
def Find_Influential_Spreaders_mercator(x,g,k):
    spreaders=[]
    gt=copy.deepcopy(g)
    nodes=copy.deepcopy(x)
    for i in range(len(nodes)):
        nodes[i,2]=g.degree(int(nodes[i,0]))
    n=len(x)
    nodes=nodes[nodes[:, 1].argsort()]
    for i in range(k):
        infnode_id=np.argmax(nodes[int(i*(n/k)):int((i+1)*(n/k)),2])
        trueid = int(nodes[int(i*(n/k))+infnode_id,0])
        spreaders.append(trueid)
        for j in gt.es:
            if trueid in j.tuple:
                gt.delete_edges(j)
        for i in range(len(nodes)):
            nodes[i,2]=g.degree(int(nodes[i,0]))
    return spreaders

#x: dictionary with community structure
#g: igraph object
#k: number of spreaders
def Find_Influential_Spreaders_community(x,g,k):

    spreaders=[]
    gt=g.copy()
    AD_nodes=g.degree()
    communities={i:[j for j in x if x[j]==i] for i in range(1,max(x.values())+1)}
    #select communities to draw from
    weights={i:len(communities[i])/g.vcount() for i in range(1,1+len(communities))}
    weights=dict(sorted(weights.items(), key=lambda item: item[1],reverse=True))
    #find max adaptive degree in community
    #select communities to draw based on their weights
    communities_selected=random.choices(list(weights.keys()),weights=list(weights.values()),k=k)
    #check that we aren't choosing more nodes from a community than its size- rare occurence, but being extra careful after recent events
    while min([len(communities[com])-communities_selected.count(com) for com in communities_selected])<0:
        communities_selected=random.choices(list(weights.keys()),weights=list(weights.values()),k=k)

    for i in communities_selected:
        node,ad=None,0
        for j in communities[i]:
            if AD_nodes[j]>ad:
                node=j
                ad=AD_nodes[j]
        if node==None:
            node=random.choice(communities[i])
        spreaders.append(node)
        communities[i].remove(node)
        #update adaptive degrees
        for j in gt.es:
            if node in j.tuple:
                gt.delete_edges(j)
        for i in range(g.vcount()):
            AD_nodes[i]=gt.degree(i)
        if len(spreaders)==k:
            return spreaders
    return spreaders

#x: dictionary with graph partitions
#g: igraph object
#k: number of spreaders
def Find_Influential_Spreaders_graph_partition(g,x,k):
    #x: dictionary with graph partitions
    #k: number of spreaders
    spreaders=[]
    gt=g.copy()
    AD_nodes=g.degree()
    communities={i:[j for j in x if x[j]==i] for i in range(1,max(x.values())+1)}
    #select communities to draw from
    #find max adaptive degree in community
    partitions=[i%(len(communities))+1 for i in range(k)]
    for i in partitions:
        node,ad=None,0
        for j in communities[i]:
            if AD_nodes[j]>=ad:
                node=j
                ad=AD_nodes[j]
        spreaders.append(node)
        #update adaptive degrees
        rmset=[]
        for e in gt.es:
            if e.tuple[0]==node or e.tuple[1]==node:
                rmset.append(e.index)
        gt.delete_edges(rmset)
        for deg in range(g.vcount()):
            AD_nodes[deg]=gt.degree(deg)
        if len(spreaders)==k:
            return spreaders
    return spreaders

#x: dictionary with graph partitions
#g: igraph object
#k: number of spreaders
#ranking: overall ranking based on desired metric
def Find_Influential_Spreaders_partition_with_ranking(x,g,k,ranking):
    spreaders=[]
    communities={i:[j for j in x if x[j]==i] for i in range(1,max(x.values())+1)}
    #select communities to draw from
    weights={i:len(communities[i])/g.number_of_nodes() for i in range(1,1+len(communities))}
    weights=dict(sorted(weights.items(), key=lambda item: item[1],reverse=True))
    #select communities to draw based on their weights
    communities_selected=random.choices(list(weights.keys()),weights=list(weights.values()),k=k)

    for i in communities_selected:
        node,ad=x,0
        for j in communities[i]:
            if ranking[j]>ad:
                node=j
                ad=ranking[j]
        spreaders.append(node)
        communities[i].remove(node)

    return spreaders
####################################################################################################

#Mercator Embedding
def Mercator_Local(networkname):
    mercator.embed('Data/networks/'+networkname+'.txt', output_name='Data/embeddings/'+networkname+'_mercator')
    x1=np.loadtxt('Data/embeddings/'+networkname+'_mercator.inf_coord')
    emb_info=[int(i[0]) for i in x1]
    if min(emb_info)==1:
        X1=np.array([np.array([int(i[0]-1),i[2],i[3]]) for i in x1])
    else:
        X1=np.array([np.array([int(i[0]),i[2],i[3]]) for i in x1])
    return X1



#Handy functions

def powerlaw_exp(g):
    D=[j for i,j in g.degree()]
    results=powerlaw.Fit(D)
    alpha= results.power_law.alpha
    return alpha

def LFR(n,mu,t1,t2,max_k,avg_k,commmunities=False,igraph_obj=False):
    N,Mu,T1,T2,maxk,k=str(n),str(mu),str(t1),str(t2),str(max_k),str(avg_k)
    s='./benchmark -N '+N+' -mu '+Mu+ ' -maxk ' +maxk  + ' -k '+k  +' -t1 ' +T1+' -t2 ' +T2
    os.system(s)
    # time.sleep(3)
    x=np.loadtxt('network.dat')
    if int(x[0][0])==1:
        edges=[(int(x[i][0])-1,int(x[i][1]-1)) for i in range(len(x))]
    else:
        edges=[(int(x[i][0])-1,int(x[i][1]-1)) for i in range(len(x))]
    g=nx.erdos_renyi_graph(n,0)
    g.add_edges_from(edges)
    if igraph_obj:
        gI = Graph(n=n, edges=list(g.edges()), directed=False)
        
        if commmunities:
            x=np.loadtxt('community.dat')
            coms={int(x[i][0])-1:int(x[i][1]) for i in range(len(x))}
            return g,gI,coms
        return g,gI
    else:
        if commmunities:
            x=np.loadtxt('community.dat')
            coms={int(x[i][0])-1:int(x[i][1]) for i in range(len(x))}
            return g
        return g

def makepartition(partition):
    out=[]
    for j in range(max(partition.keys())):
        out.append({i for i in partition if partition[i]==j})
    return out

def get_modularity_assortativity(g):
    g=nx.convert_node_labels_to_integers(g)
    partition=community_louvain.best_partition(g)
    return (nx_comm.modularity(g,makepartition(partition)),nx.degree_assortativity_coefficient(g))

def degree_heterogeneity(g):
    d=[j for i,j in g.degree()]
    d2=[j**2 for j in d]
    return np.mean(d2)/(np.mean(d))**2


#  greedy protocol

def greedy_map(gI,greedy_visited):
    distances=[]
    for i in range(len(greedy_visited)-1):
        print(gI.get_shortest_paths(greedy_visited[i],greedy_visited[i+1],mode='all'))
        distances.append(len(gI.get_shortest_paths(greedy_visited[i],greedy_visited[i+1],mode='all')[0])-1)
    return distances

def GreedyChen(g,p,s):
    S=[]
    R=2000
    nodetoclus,clusters=GetInstances(g,p,R)
    for i in range(s):
        sv=ComputeScore(g,S,nodetoclus,clusters,R)
        max_value = max(sv.values())
        max_keys = [k for k, v in sv.items() if v == max_value]
        best=int(np.random.choice(max_keys,1))
        S.append(best)
        for n in nodetoclus[best]:
            clusters[n]=0
    return S

def ComputeScore (g,S,nodetoclus,clusters,R):
    sv=dict(zip(list(g.nodes()),np.zeros(g.number_of_nodes())))
    for n in S:
        sv[n]=-1
    for j in nodetoclus:
        for x in nodetoclus[j]:
            sv[j]+=clusters[x]
    return sv

def GetInstances(g,p,R):
    N=g.number_of_nodes()
    Nodes=dict(zip(list(g.nodes()),[[] for n in range(N)]))
    AllClusters=dict()
    for i in range(R):
        lb=N*i+1
        ub=N*(i+1)+1
        nodetoclus=dict(zip(list(g.nodes()),range(lb,ub)))
        clusters=dict(zip(range(lb,ub),[[j] for j in list(g.nodes())]))
        #remove each edge w.p. 1-p
        k=int(np.random.binomial(g.number_of_edges(),p,1))
        edges=random.sample(g.edges(),k)
        nodetoclus,clusters=GraphClusters(edges,nodetoclus,clusters)
        for i in list(g.nodes()):
            Nodes[i].append(nodetoclus[i])
        #set cluster values to cluster sizes
        for c in clusters:
            clusters[c]=len(clusters[c])
        AllClusters.update(clusters)
    return Nodes,AllClusters

def GraphClusters(edges,nodetoclus,clusters):
    for e1,e2 in edges:
        c1=nodetoclus.get(e1)
        c2=nodetoclus.get(e2)
        if c1!=c2:
            if len(clusters[c1]) >= len(clusters[c2]):
                for n1 in clusters[c2]:
                    nodetoclus[n1]=c1
                clusters[c1]+=clusters[c2]
                del clusters[c2]
            else:
                for n1 in clusters[c1]:
                    nodetoclus[n1]=c2
                clusters[c2]+=clusters[c1]
                del clusters[c1]
    return nodetoclus, clusters

def Greedy_IC(g,p,s,name,rep=1,directory='Data/greedy/'):
    rankings=[]
    for i in range(rep):
        rankings.append(GreedyChen(g,p,s))
    return rankings[0]

#centrality metrics


def CI1(g,ensembles):
    
    
    """
    input - g -> networkx graph object (unweighted, undirected) 
            k -> int -> number of seeds required
            ensembles -> int -> number of runs of the ftm on g
            
    Output: - ranked CI1 for all nodes
    """
      
    # --------------------
    # Find the values of CI-1 for all nodes
    # --------------------
    
#     start_time = time.time() 
    # Create the sorted list of nodes and their CI
    ki,kj,CIs={},{},{}
    for node in g.nodes:
        ki[node]=g.degree(node)-1
    for node in g.nodes:
        kj[node]= np.sum([ki[x] for x in g.neighbors(node)])
        CIs[node]=ki[node]*kj[node]
        sortedCIs = dict(sorted(CIs.items(), key=lambda item: item[1],reverse =True))
    
    S = list(sortedCIs.keys())
    return S,CIs

def CI2(g,ensembles=1):  
    """
    input - g -> networkx graph object (unweighted, undirected) 
            k -> int -> number of seeds required
            ensembles -> int -> number of runs of the ftm on g
            
    Output: -ranked CI2 for all nodes
    """
      
    # --------------------
    # Find the values of CI-2 for all nodes
    # --------------------
    
#     start_time = time.time() 
    # Create the sorted list of nodes and their CI
    CIs={}
    secondngbs={}
    d={i:j for i,j in g.degree()}
    for node in g.nodes:
        x=[]
        for ng in g.neighbors(node):
            for sng in g.neighbors(ng):
                if sng!=node and sng not in x:
                    x.append(sng)
        kj=sum([d[node] for i in x])
        CIs[node]=(d[node]-1)*(kj-len(x))
    sortedCIs = dict(sorted(CIs.items(), key=lambda item: item[1],reverse =True))
    S = list(sortedCIs.keys())
    return S,CIs

def LocalRank(g,ensembles):
    """
    input - g -> networkx graph object (unweighted, undirected) 
            k -> int -> number of seeds required
            ensembles -> int -> number of runs of the ftm on g
            
    Output: -ranked Local Rank for all nodes
    """
    Q={}
    CL={}
    for node in list(g.nodes()):
        NeighList=list(g.neighbors(node))
        tempList=[]
        for neigh in NeighList:
            tempList.extend(list(g.neighbors(neigh)))
        NeighList=NeighList+tempList
        NeighList=list(set(NeighList))
        NeighList.remove(node)
        Q[node]=len(NeighList)
    for node in list(g.nodes()):
        CL[node]=sum([Q[a] for a in list(g.neighbors(node))])
    
    sortedCL = dict(sorted(CL.items(), key=lambda item: item[1],reverse =True))
    
    S = list(sortedCL.keys())
    return S,CL

def nonbacktracking(g,ensembles):
    N=len(g)
    g_temp=nx.relabel_nodes(g,dict(zip(list(g.nodes()),list(range(N)))))
    list12=[]
    for n in list(g_temp.nodes()):
        list12.append(1-g_temp.degree(n))
    data=[1]*(2*len(g_temp.edges()))
    i=[e[0] for e in g_temp.edges()]+[e[1] for e in g_temp.edges()]
    j=[e[1] for e in g_temp.edges()]+[e[0] for e in g_temp.edges()]

    data.extend(list12)
    i.extend(list(range(N)))
    j.extend(list(range(N,2*N)))

    data.extend([1]*N)
    i.extend(list(range(N,2*N)))
    j.extend(list(range(N)))

    data=np.array(data)
    i=np.array(i)
    j=np.array(j)

    theMatrix=sparse.coo_matrix((data,(i,j)))

    NB=dict(zip(list(g.nodes()),np.absolute(sparse.linalg.eigs(theMatrix.asfptype(),k=1)[1][:N])))
    
    for key in NB:
        NB[key]=float(NB[key])
        
    sortedNB = dict(sorted(NB.items(), key=lambda item: item[1],reverse =True))
    
    S = list(sortedNB.keys())
    return S,sortedNB

def hindex(g,ensembles=1):
    HI = {}
    for n in g.nodes():
        ngb=[g.degree(m) for m in g.neighbors(n)]
    # iterating over the list
        for i in range(len(ngb),0,-1):
            # if result is less than or equal
            # to cited then return result
            temp=True
            for node in ngb:
                if node<i:
                    temp=False
            if temp:
                HI[n] = i
                break
    sortedHI = dict(sorted(HI.items(), key=lambda item: item[1],reverse =True))
    return list(sortedHI.keys()),HI


def betweenness(g,ensembles):
    x=nx.betweenness_centrality(g)
    return list(dict(sorted(x.items(), key=lambda item: item[1],reverse =True))),x

def closeness(g,ensembles):
    x=nx.closeness_centrality(g)
    return list(dict(sorted(x.items(), key=lambda item: item[1],reverse =True))),x

def eigenvector(g,ensembles):
    x=nx.eigenvector_centrality_numpy(g)
    return list(dict(sorted(x.items(), key=lambda item: item[1],reverse =True))),x

def katz(g,ensembles):
    x=nx.katz_centrality_numpy(g)
    return list(dict(sorted(x.items(), key=lambda item: item[1],reverse =True))),x

def pagerank(g,ensembles):
    x=nx.pagerank(g)
    return list(dict(sorted(x.items(), key=lambda item: item[1],reverse =True))),x

def degree(g,ensembles):
    x=nx.degree_centrality(g)
    return list(dict(sorted(x.items(), key=lambda item: item[1],reverse =True))),x

def adaptive_degree(G,ensembles=1):
    g=G.copy()
    AD={}
    for _ in range(g.number_of_nodes()):
        deg = {a:b for a,b in g.degree()}
        AD[max(deg, key=deg.get)]=deg[max(deg, key=deg.get)]
        g.remove_node(max(deg, key=deg.get))
    return list(dict(sorted(AD.items(), key=lambda item: item[1],reverse =True))),AD


def rb(g,ensembles):
    return random.sample(list(range(0, g.number_of_nodes()+1)),g.number_of_nodes()),random.sample(list(range(0, g.number_of_nodes()+1)),g.number_of_nodes())

def kshell(g,ensembles=1):
    G=g.copy()
    k_list = []
    c=0
    while len(k_list)<G.number_of_nodes():
        x=list(nx.k_shell(G,c).nodes)
        for j in x:
            k_list.append(j)
        c+=1
    k_list=k_list[::-1]
    return k_list,k_list
