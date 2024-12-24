import sys
import json
import numpy as np
# To store in COO format
import scipy.sparse as sp
import matplotlib.pyplot as plt

def parse_to_matrix_format(filename):
    """
    We need to output the graph having it in an node-arc incident matrix format + the flow of each node

    For a node with id, flow 

    We save a diagonal nxn matrix with the flow of each node in the diagonal

    For an edge with src, dst, low, cap, cost we save a nxm node-arc incident matrix with the following rules:
    - If the edge is outgoing from v->w then we get negative cost 
    - If the edge is incoming from v<-w then we get positive cost

    """
    problem = {}

    flows = []
    edges = []
    full_edges = []
    
    counter = 0

    with open(filename, 'r', encoding='UTF-8') as file:
        while line := file.readline():
            splat = line.strip().split(" ")
            command = splat[0]
            if (command == "p"):
                problem["type"] = splat[1]
                problem["n_nodes"] = int(splat[2])
                problem["m_arcs"] = int(splat[3])
            if (command == "n"):
                # We have for a node: id, flow
                flows.append((int(splat[1]), int(splat[2])))
                counter += 1
            if (command == "a"):
                # add in position src,dst of nxm matrix the cost of the edge
                src = int(splat[1])
                dst = int(splat[2])
                low = int(splat[3])
                cap = int(splat[4])
                cost = int(splat[5])
                
                edges.append((src, dst, cost))
                full_edges.append((src, dst, low, cap, cost))
            if (command == "c"):
                # comment
                # Could be used to store the problem setting
                pass

    node_flows = [0] * problem["n_nodes"]

    for f in flows:
        node_flows[f[0]-1] = f[1]
    
    # Initialize an empty matrix
    node_arc_matrix = np.zeros((problem["n_nodes"], len(edges)))
    
    # Populate the node-arc incident matrix
    for idx, (src, dst, cost) in enumerate(edges):
        node_arc_matrix[src-1][idx] = -1  # outgoing edge
        node_arc_matrix[dst-1][idx] = 1   # incoming edge
         
    # DO NOT SAVE AS OBJECTS node_arc_matrix_sparse 
    
    # Save the two matrixes in a single file, the file will be opened in julia
    np.savez_compressed(f'{filename}_out.npz', node_flows=node_flows,
                        node_arc_matrix=node_arc_matrix,
                        full_edges=full_edges)
    
    pass 

if __name__ == "__main__":
    filename = sys.argv[1]
    #TODO: Parse a list of file names
    parse_to_matrix_format(filename)

    
