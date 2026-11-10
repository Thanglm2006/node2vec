import numpy as np

def random_walk(G,start,walk_length,alias_edges):
    walk = [start]
    for i in range(walk_length):
        current = walk[-1]
        candidates = list(G.neighbors(current))
        if len(walk)==1:
            weights= np.array([ G[current][x].get('weight',1.0) for x in candidates ])
            probs=weights/np.sum(weights)
        else:
            previous = walk[-2]
            trans=alias_edges[(previous,current)]
            candidates= list(trans["candidates"])
            probs=trans["probs"]
        next_node=np.random.choice(candidates,p=probs)
        walk.append(next_node)
    return walk


