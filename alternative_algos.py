import networkx as nx
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm
import cProfile, pstats, io, random


#---------------PREPROCESSING------------------

graph = nx.DiGraph()
# for each link in links add edge
# the weight of edges is 1 (for all)
for line in open('wikigraph_reduced.csv', 'r').readlines()[1:]:
    nodes = line.replace('\n', '').split('\t')
    graph.add_edge(int(nodes[1]), int(nodes[2]), weight=1)


categories = dict()
#for each category set the name as key and a list of nodes as article
for line in open('wiki-topcats-categories.txt', 'r').readlines():
    s = line.index(':')
    e = line.index(';')
    cat_name = line[s+1 : e].replace('_', ' ')
    cat_list = line[e+1:].split()
    categories[cat_name] = [int(c) for c in cat_list]


# inverted idx, for each value all its categories
art_cat = defaultdict(list)
for name, values in categories.items():
    for value in values:
        art_cat[value].append(name)

# keep 1 cat per node
page_category = { page : np.random.choice(values, 1)[0] for page, values in art_cat.items() if page in graph.nodes }

# overwrite categories
categories = defaultdict(list)
for page, category in page_category.items():
    categories[category].append(page)


# get ordered page_names
page_names = list()
for line in open('wiki-topcats-page-names.txt', 'r').readlines():
    i = line.index(' ')
    page_names.append(line[i+1:].replace('\n', ''))


# enumerate categories
inverted_page_names = dict()
for index, name in enumerate(page_names):
    inverted_page_names[name] = index

#------------------------ RQ2 -----------------------------

def exploring(v, d):
    """Takes a page name and the number of clicks, returns all the pages that are reachable within those number of clicks

    Args:
        v (str): name of the page
        d (int): number of clicks available

    Returns:
        [set]: distinct reachable pages
    """
    v = page_names.index(v)
    i = 0
    neighbors = [v]
    indices = [v]

    while i < d:
        prev_neighbors = neighbors
        neighbors = []
        for x in prev_neighbors:
            neighbors.extend(graph.neighbors(x))
        indices.extend(neighbors)
        i += 1

    return { page_names[i] for i in indices }

#print(exploring("Marty O'Brien", 2))


#--------------------------RQ4----------------------


# finds shortest path between 2 nodes of a graph using BFS
def bfs_shortest_path(u, v, g):
    # u: start node
    # v: goal node
    # g: graph
    
    # keep track of explored nodes
    explored = []
    # keep track of all the paths to be checked
    queue = [[u]]
 
    # return path if start is goal
    if u == v:
        return "Exception: u and v are the same node" # no path! the nodes are the same
 
    # keeps looping until all possible paths have been checked
    while queue:
        # pop the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        if node not in explored:
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in g.neighbors(node):
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # return path if neighbour is goal
                if neighbour == v:
                    return new_path
 
            # mark node as explored
            explored.append(node)
 
    # in case there's no path between the 2 nodes
    return "Not possible"


def my_double_cat_subg(c1, c2, g):
    # c1: category 1
    # c2: category 2
    cat_nodes = set(categories[c1]) | set(categories[c2]) # the set of all possible nodes
    cat_nodes = cat_nodes & set(g.nodes) # clean the node
    sg = nx.DiGraph()
    sg.add_nodes_from((n, g.nodes[n]) for n in cat_nodes)
    sg.add_edges_from((n, nbr, d) for n, nbrs in g.adj.items() if n in cat_nodes\
                      for nbr, d in nbrs.items() if nbr in cat_nodes)
    return sg


def max_flow(source, sink, g):
    """returns max_flow between two nodes

    Args:
        source (int): starting node
        sink (int): end node
        g (networkx graph): graph
    """
    max_flow = 0
    path = bfs_shortest_path(source, sink, g)
    
    while not isinstance(path, str):
        for i in range(1, len(path)):
            g.remove_edge(path[i-1], path[i])
        max_flow += 1
        path = bfs_shortest_path(source,sink,g)
    
    return max_flow

# cat_1 = 'Southampton F.C. players'
# cat_2 = 'English footballers'
# # sg = my_double_cat_subg('Southampton F.C. players','English footballers', graph)
# u = random.choice(categories[cat_1])
# v = random.choice(categories[cat_2])

#print(max_flow(u,v,graph))


#------------------------RQ5-----------------------------

def profile(fnc):

    """a decorator that uses Cprofile to profile a function"""

    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc (*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def compute_node_distances(u, g):
    """Uses BFS to compute all the distances from a node
    Args:
        u (int): starting node number
        g (Networkx graph): graph

    Returns:
        [dict]: mapping from node number to distance from u
    """
    # keep track of explored nodes and their distances
    explored = {}    
    explored[u] = 0
    # keep track of all the paths to be checked
    queue = [u]
    # keeps looping until all possible paths have been checked
    while queue:
        # pop the first path from the queue
        node = queue.pop(0)
            # go through all neighbour nodes, store their distances
            # push them into the queue
        for neighbour in g.neighbors(node):
            if neighbour not in explored:
                explored[neighbour] = explored[node] + 1
                queue.append(neighbour)
    return explored


def compute_all_nodes_distances(category, graph):
    """Computes all the distances from each node in a category

    Args:
        category (list): list of node numbers in a category
        graph (Networkx graph): graph

    Returns:
        [dict of dicts]: mapping from all nodes in a category to all the distances from that node
                         e.g. dist[n1][n2] is the distance from node n1 to node n2

    TODO: convert dict of dicts to single dict with tuple (n1, n2) as a key, is it worth it? 
    """
    dist = {}
    for n1 in category:
        dist[n1] = compute_node_distances(n1, graph)
    return dist

#@profile
def sort_categories_by_distance(c, categories, graph):
    """Takes a category, returns the other categories sorted by their distance from the first one (closest first).
       The distance is defined as the median of the distances between each node of the first category and the others

    Args:
        c (str): name of the starting category
        categories (dict): mapping from all categories names to their nodes
        graph (Networkx graph): graph

    Returns:
        [list of tuple]: sorted categories and their distance
                         e.g. [(cat2, 3), (cat3, 5)]
    """
    other_cat = categories.copy()
    del other_cat[c]
    distances = compute_all_nodes_distances(categories[c], graph)
    medians = []

    for cat in tqdm(other_cat):
        nodes = categories[cat]
        my_list = []
        for node in nodes:
            for i in categories[c]:
                if node in distances[i]:
                    my_list.append(distances[i][node])
                else:
                    my_list.append(float("inf"))
        medians.append((cat,np.median(np.array(my_list))))

    return sorted(medians, key = lambda x: x[1])


if __name__ == "__main__":

    # RQ2

    #print(exploring("Marty O'Brien", 2))

    # RQ4

    # cat_1 = 'Southampton F.C. players'
    # cat_2 = 'English footballers'
    # sg = my_double_cat_subg(cat_1, cat_2, graph)
    # u = random.choice(categories[cat_1])
    # v = random.choice(categories[cat_2])
    # print(max_flow(u,v,graph))

    # RQ5

    print(sort_categories_by_distance('English footballers', categories, graph))