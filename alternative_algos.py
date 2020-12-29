import networkx as nx
from collections import defaultdict

from networkx.classes.function import neighbors

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

# get ordered page_names
page_names = list()
for line in open('wiki-topcats-page-names.txt', 'r').readlines():
    i = line.index(' ')
    page_names.append(line[i+1:].replace('\n', ''))


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

print(exploring("Marty O'Brien", 2))