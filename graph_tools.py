import numpy as np
import itertools
from scipy.special import comb
from scipy.spatial.distance import squareform
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import math
import string
import sklearn.cluster as cluster
from sklearn.decomposition import NMF

"""
Code for defining symmetry groups (motifs) in small undirected subgraphs, finding those motifs within a larger undirected graph, and generating graphs that are composed of these motifs.

	>>> G = nx.fast_gnp_random_graph(12, .5)
	>>> relations, motifs = gen_relational_tensor(G, motif_size=4)
	>>> draw_graph_motifs(G, relations, motifs)
"""

def draw_graph(G, node_size=200, node_labels=True, size=3, font_size=15, font_color='white'):
	plt.figure(figsize=(size,size))
	pos = nx.drawing.layout.circular_layout(G)
	nx.draw(G, pos, node_size=node_size, with_labels=node_labels, font_size=font_size, font_color=font_color)

def gen_adj_mats(motif_size):
	"""
	Returns all symmetric n by n adjacency matrices.

	Parameters
	----------
	motif_size: int
		size of the adjacency matrices

	Returns
	-------
	adj_mats: lst
		List of all n by n symmetric adjacency matrices.
	"""
	nodes = np.arange(motif_size)
	configs = [tup for tup in itertools.product([0, 1], repeat=int(comb(len(nodes), 2)))]
	configs = sorted(configs, key=lambda x: sum(x))
	adj_mats = []
	for config in configs:
		t = squareform(config)
		if np.count_nonzero(np.sum(t, axis=0)) >= motif_size and np.count_nonzero(np.sum(t, axis=1)) >= motif_size:
			if motif_size == 2 or not np.array_equal(np.sum(t, axis=0), np.ones(motif_size)):
				adj_mats.append(t)
	return adj_mats

def transform(adj_mats):
	"""
	Returns a list containing all permutations of the adjacency
	matrices in adj_mats.

	Parameters
	----------
	adj_mats: lst
		Output of gen_adj_mats

	Returns
	-------
	transforms: lst
		List the same length as adj_mats, where transforms[i] is a
		list containing all permutations of adj_mats[i]
	"""
	n = np.arange(len(adj_mats[0]))
	perm = list(itertools.permutations(n))
	perm_rules = [list(zip(n, i)) for i in perm]
	transforms = []
	for mat in adj_mats:
		mat_transforms = []
		for rule in perm_rules:
			transform = mat.copy()
			for tup in rule:
				transform[:, tup[0]] = mat[:, tup[1]]
			ref = transform.copy()
			for tup in rule:
				transform[tup[0], :] = ref[tup[1], :]
			mat_transforms.append(transform)
		transforms.append(mat_transforms)
	return transforms

def gen_perm_mats(edge_size):
	"""
	Returns a list containing all permutation edge_size x edge_size permutation matrices.

	Parameters
	----------
	edge_size: int
		Desired edge size for the permutation matrices

	Returns
	-------
	perm_mats: lst
		List of all edge_size x edge_size permutation matrices.
	"""
	n = np.arange(edge_size)
	perm = list(itertools.permutations(n))
	perm_rules = [list(zip(n, i)) for i in perm]
	perm_mats = []
	mat = np.identity(edge_size)
	for rule in perm_rules:
		perm_mat = mat.copy()
		for tup in rule:
			perm_mat[:, tup[0]] = mat[:, tup[1]]
		perm_mats.append(perm_mat)
	return perm_mats

def group_motifs(motif_size):
	"""
	Groups all non-redundant permutations of adj_mats. Returns a list containing the motif groups.

	Parameters
	----------
	motif_size: int
		Size of the motifs

	Returns
	-------
	motifs: lst
		List of motifs. motifs[i] indexes motif i and contains every permutation of motif i.
	"""
	adj_mats = gen_adj_mats(motif_size)
	transforms = transform(adj_mats)
	match = np.zeros((len(adj_mats), len(adj_mats)))
	for i, mat_1 in enumerate(adj_mats):
		for j, mat_2 in enumerate(transforms):
			n = len([x for x in mat_2 if (x == mat_1).all()])
			if n > 0:
				match[i, j] = 1
	m = [list(np.nonzero(x)[0]) for x in match]
	m.sort()
	m = list(motifs for motifs,_ in itertools.groupby(m))
	motifs = [[adj_mats[i] for i in n] for n in m]
	return motifs

def gen_relational_tensor(graph, motif_size):
	"""
	Returns a motif_size-D tensor, where the length of each dimension is equal to the number of nodes in graph. relations[a, b, ... , n] indexes an ordered motif_size subgraph (a < b < ... < n). The value of relations[a, b, ... , n] is an int which indexes motifs if that motif[i] holds for that subgraph, or 0 otherwise. Also returns motifs, the list of motif groups.

	Parameters
	----------
	graph: nx.Graph
	motif_size: Size of motifs to search for.

	Returns
	-------
	relations: motif_size-D arr
		Tensor encoding which motifs hold for which ordered subgraphs of graph.
	motifs: lst
		List of motif groups. Output of group_motifs.
	"""
	motifs = group_motifs(motif_size)
	tensor_shape = tuple(np.repeat(len(graph.nodes()), motif_size))
	relations = np.zeros(tensor_shape)
	it = np.nditer(relations, flags=['multi_index'])
	while not it.finished:
		if not len(set(it.multi_index)) < len(it.multi_index): #no self-relations
			subgraph = graph.subgraph(list(it.multi_index))
			adj_mat = nx.adjacency_matrix(subgraph).todense()
			for idx, motif in enumerate(motifs):
				for transformation in motif:
					if (adj_mat == transformation).all():
						relations[it.multi_index] = idx
		it.iternext()
	return relations, motifs

def draw_graph_motifs(graph, relations, motifs):
	"""
	Plots graph and saves a .png for every motif, where the motif is colored within the larger graph. Takes a graph and the output of gen_relational_tensor as input.
	"""
	all_motifs = []
	for idx, motif in enumerate(motifs):
		motif_nodes = []
		it = np.nditer(relations, flags=['multi_index'])
		while not it.finished:
			if it[0] == idx and not len(set(it.multi_index)) < len(it.multi_index):
				motif_nodes.append(it.multi_index)
			it.iternext()
		all_motifs.append(motif_nodes)
	for i, motif in enumerate(all_motifs):
		for j, nodes in enumerate(motif):
			motif[j] = tuple(sorted(nodes))
			motif.sort()
			all_motifs[i] = list(k for k,_ in itertools.groupby(motif))
	for i, motif in enumerate(all_motifs):
		if i is not 0:
			for j, nodes in enumerate(motif):
				subgraph_edges = graph.subgraph(list(nodes)).edges()
				edge_colors = []
				edge_widths = []
				for edge in graph.edges():
					if edge in subgraph_edges:
						edge_colors.append(40 + 5*i)
						edge_widths.append(2)
					else:
						edge_colors.append(0)
						edge_widths.append(1)
				node_colors = []
				node_size = []
				for node in graph.nodes():
					if node in nodes:
						node_colors.append(40 + 5*i)
						node_size.append(30)
					else:
						node_colors.append(0)
						node_size.append(10)
				pos = nx.drawing.layout.circular_layout(graph)
				edge_vmax=40 + 5 * len(all_motifs)
				cmap = matplotlib.cm.get_cmap('OrRd')
				norm = matplotlib.colors.Normalize(vmin=0, vmax=edge_vmax)
				fig = plt.figure()
				fig.suptitle('motif '+str(i), fontsize=14, color=cmap(norm(40 + 5*i)))
				nx.draw(graph, edge_cmap = plt.cm.OrRd, edge_vmin=0, edge_vmax=edge_vmax, cmap = plt.cm.OrRd, vmin=0, vmax=edge_vmax, node_color=node_colors, edge_color=edge_colors, pos=pos, width=edge_widths, node_size=node_size)
				fig.patch.set_facecolor('#D1D1D1')
				fig.savefig('motif_'+str(i)+'_'+str(j)+'.png', facecolor=fig.get_facecolor())
				plt.close()

def plot_motifs(motifs):
	"""
	Saves a .png of each motif group. Each permutation of a motif is plotted on the same axis.
	"""
	for motif_n, idxs in enumerate(motifs):
		N = len(idxs)
		cols = 4
		rows = int(math.ceil(N / cols))
		gs = gridspec.GridSpec(rows, cols)
		fig = plt.figure(figsize=(4, rows))
		for n, idx in enumerate(idxs):
		    ax = fig.add_subplot(gs[n])
		    do_plot(idx, ax)
		plt.savefig('plots/motif_'+str(motif_n)+'.png')
		plt.close

def do_plot(idx, ax):
	"""
	Helper function for plot_motifs.
	"""
	G = nx.from_numpy_matrix(idx)
	pos = nx.drawing.layout.circular_layout(G)
	nx.draw(G, pos, ax, node_size=20)

def gen_graphs_batch(n_graphs, n_motifs=20, motif_sizes=[2, 3, 4], prior=np.array([1, 1, 5, 7, 1, 1, 1, 7, 7])/31):
	"""
	Returns a list of networkx graphs generated according to a distribution over motifs.

	Parameters
	----------
	n_graphs: int
		Number of graphs to generate.
	n_motifs: int
		Number of motifs to place in the graph.
	motif_size: lst
		A list of motif sizes, where all (non-degenerate) n-node motifs will be considered for the n's listed.
	prior: np.array
		An array that encodes the probability of drawing each unique motif. This is the length of the number of unique motifs. For motif_sizes=[2, 3, 4], the length should be 9.

	Returns
	-------
	graphs: lst
		List of networkx graphs.
	motifs: dic
		Dictionary of motifs used to generate the graphs.
	"""
	motifs_clust = {m: group_motifs(m) for m in motif_sizes}
	motif_tups = [(m, n) for m in motif_sizes for n in range(len(motifs_clust[m]))]
	motifs = {m: n for m, n in zip(motif_tups, [j for i in motifs_clust for j in motifs_clust[i]])}
	motif_colors = {m: n for m, n in zip(motif_tups, range(1, len(motif_tups)+1))}
	graphs = []
	for i in range(n_graphs):
		G = nx.Graph()
		rand_motifs = [motif_tups[i] for i in np.random.choice(np.arange(0, len(motif_tups)), size=n_motifs, p=prior)]
		for motif in rand_motifs:
			idx = np.random.randint(len(motifs[motif])) #permutation is drawn from random uniform
			m = nx.from_numpy_matrix(motifs[motif][idx])
			orig_nodes = np.arange(motif[0])
			new_nodes = np.random.randint(len(G.nodes())+motif[0]+1, size=motif[0])
			m = nx.relabel_nodes(m, {x: y for x, y in zip(orig_nodes, new_nodes)})
			nx.set_node_attributes(m, 'motifs', [motif])
			nx.set_edge_attributes(m, 'motif', motif)
			# nx.set_node_attributes(m, 'neighbors', [tuple(new_nodes)])
			nx.set_node_attributes(m, 'color', motif_colors[motif])
			nx.set_edge_attributes(m, 'color', motif_colors[motif])
			for j in m.nodes():
			    if j in G.nodes():
			        nx.set_node_attributes(m, 'motifs', {j: nx.get_node_attributes(m, 'motifs')[j] + nx.get_node_attributes(G, 'motifs')[j]})
			        # nx.set_node_attributes(m, 'neighbors', {j: nx.get_node_attributes(m, 'neighbors')[j] + nx.get_node_attributes(G, 'neighbors')[j]})
			G = nx.compose(G, m)
		G = nx.convert_node_labels_to_integers(G)
		graphs.append(G)
	return graphs, motifs

def laplacians(graphs):
	"""
	Generates normalized laplacian matrices for a list of graphs.
	"""
	Ls = [nx.normalized_laplacian_matrix(G).todense() for G in graphs]
	return Ls

def spectral_cluster_and_plot(G, n_clusters=3):
	"""
	Performs spectral clustering and plots the graph with nodes colored according to cluster.
	"""
	c = cluster.SpectralClustering(n_clusters)
	adj_mat = nx.adjacency_matrix(G).todense()
	c.fit(adj_mat)
	predictions = c.fit_predict(adj_mat)
	pos = nx.drawing.layout.spectral_layout(G)
	nx.draw(G, pos=pos, node_color=predictions)
	return predictions

def association_graph(G1, G2):
	a1 = nx.adjacency_matrix(G1).todense()
	a2 = nx.adjacency_matrix(G2).todense()
	edges = list(itertools.product(range(len(G1)), range(len(G1))))
	association_matrix = np.zeros((len(edges), len(edges)))
	for i, e1 in enumerate(edges):
		for j, e2 in enumerate(edges):
			if e1[0] != e2[0] and e1[1] != e2[1]:
				association_matrix[i, j] = 1 - (a1[e1[0], e2[0]] - a2[e1[1], e2[1]]) ** 2
	labels = []
	for e in edges:
		labels.append(list(G1.nodes())[e[0]] + list(G2.nodes())[e[1]])
	association_graph = nx.from_numpy_matrix(association_matrix)
	association_graph = nx.relabel_nodes(association_graph, {x: labels[x] for x in range(len(association_graph))})
	return association_graph, association_matrix, labels



def plot_generated_graph(G, motifs):
	"""
	Takes a generated graph and the set of motifs that generated as input.

	Generates three plots:
	g_plt:
		A plot of the graph, with edges and nodes color-coded by motif type.
	am_plt:
		A plot of the graph's adjacency matrix, with transitions color-coded by motif type.
	perm_plt:
		A plot showing the adjacency matrix of every permutation of every motif + the graph of that motif. Motifs are color-coded.

	Can be altered later to save to .png
	"""
	motif_colors = {m: n for m, n in zip(motifs.keys(), range(1, len(motifs)+1))}
	cmap = plt.cm.gist_ncar
	pos = nx.drawing.layout.circular_layout(G)
	g_plt = nx.draw(G, pos, node_size=30, width=3, node_color=list(nx.get_node_attributes(G,'color').values()), cmap=cmap, edge_color=list(nx.get_edge_attributes(G,'color').values()), edge_cmap=cmap, vmin=0, vmax=len(motif_colors))

	adj_mat = nx.adjacency_matrix(G).todense()
	adj_mat_coded = np.zeros(adj_mat.shape)
	for i, node1 in enumerate(G.nodes()):
	    for j, node2 in enumerate(G.nodes()):
	        if (node1, node2) in G.edges():
	            adj_mat_coded[i, j] = motif_colors[nx.get_edge_attributes(G, 'motif')[(node1, node2)]]
	            adj_mat_coded[j, i] = motif_colors[nx.get_edge_attributes(G, 'motif')[(node1, node2)]]
	am_plt = plt.matshow(adj_mat_coded, cmap=cmap, vmin=0, vmax=len(motif_colors))
	plt.colorbar()

	"""For sparse coding"""
	dictionary = [motifs[m][0] for m in motifs]
	dic = [np.zeros((8, 8)) for i in dictionary]
	for i, mat in enumerate(dictionary):
	    dic[i][0:mat.shape[0], 0:mat.shape[1]] = mat
	dic = {x: y for x, y in zip(motifs.keys(), dic)}
	dictionary = {x: y for x, y in zip(motifs.keys(), dictionary)}
	""""""

	max_perms = np.max([len(dictionary[m]) for m in motifs])
	perm_plt, axes = plt.subplots(nrows=len(motifs), ncols=max_perms+1, figsize=(40, 8))

	for i, motif in enumerate(motifs):
	    for j in range(max_perms):
	        if j <= len(motifs[motif])-1:
	            mat = axes[i][j].matshow(motifs[motif][j] * motif_colors[motif], cmap=cmap, vmin=0, vmax=len(motif_colors))
	        else:
	            mat = axes[i][j].matshow(np.zeros((4, 4)), cmap=cmap, vmin=0, vmax=len(motif_colors))
	        axes[i][j].xaxis.set_visible(False)
	        axes[i][j].yaxis.set_visible(False)
	    graph = nx.draw(nx.from_numpy_matrix(motifs[motif][0]), ax=axes[i][max_perms], node_size=10)
