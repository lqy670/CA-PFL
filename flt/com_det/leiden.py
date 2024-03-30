import leidenalg
import igraph as ig
import cairocffi
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def test2(li,num):
    return np.random.choice(li,num,replace=False)

if __name__ == '__main__':
    # G = ig.Graph.Famous('Zachary')
    # part = leidenalg.find_partition(G,leidenalg.ModularityVertexPartition)
    # # ig.plot(part,'./leiden.png')
    # np.random.seed(8)
    # for i in part:
    #     print(i)
    # print(part)
    # print(part.modularity)

    # cluster_indices = [[i for i in range(10)]]
    # print(cluster_indices)
    # cluster_indices = [[client for client in range(10)]]
    # print(cluster_indices)
    # np.random.seed(22)
    # L = [[1,2,3],[2,3,4]]
    # dict = {1:3,2:4}
    # re = np.random.choice(L,5,replace=False)
    # print(re)
    # print([value for key,value in dict.items()])
    G  = nx.Graph()
    G.add_weighted_edges_from([('R','Y',7.0),('R','B',11),('R','Y',5.0)])
    pos = nx.spring_layout(G)
    nx.draw(G,pos)
    edge_labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels)
    nx.draw_networkx_labels(G,pos,alpha=0.5)
    plt.savefig('graph.png')
    plt.show()