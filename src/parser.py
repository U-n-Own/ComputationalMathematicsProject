import sys
import json

if len(sys.argv) == 1:
    print("insert filename")
    exit(1)

filename = sys.argv[1]

# TODO output adj matrix instead of json

problem = {}
graph = {"nodes" : [], "edges" : []}

with open(filename, 'r', encoding='UTF-8') as file:
    while line := file.readline():
        splat = line.strip().split(" ")
        command = splat[0]
        if (command == "p"):
            problem["type"] = splat[1]
            problem["n_nodes"] = int(splat[2])
            problem["n_arcs"] = int(splat[3])
        if (command == "n"):
            graph["nodes"].append({"id" : int(splat[1]), "flow" : int(splat[2])})
        if (command == "a"):
            graph["edges"].append({"src" : int(splat[1]), "dst" : int(splat[2]), "low" : int(splat[3]), "cap" : int(splat[4]), "cost" : int(splat[5])})

F = open(f'{filename}_out.json','w')
F.write(json.dumps({"problem" : problem, "graph" : graph}))
F.close()
