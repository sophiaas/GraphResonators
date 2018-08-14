import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_dynamics(association_matrix):
	x = np.ones(len(association_matrix)) / len(association_matrix)
	x += np.random.rand(len(x)) / 1e-17
	x /= np.sum(x)
	xs = [x.copy()]
	diff = 1
	i = 1
	while diff > 1e-10:
		x_hat = replicator(x, association_matrix)
		diff = np.sum((x - x_hat) ** 2)
		x = x_hat
		xs.append(x)
		i += 1
	print('{} iterations'.format(i))
	return xs, x

def replicator(x, association_matrix):
	pi = np.dot(association_matrix, x)
	x_hat = x * pi / np.dot(x, pi)
	return x_hat

def plot_resonator_dynamics(xs, graph):
	time = np.hstack([[x]*len(xs[0]) for x in range(len(xs))])
	mapping = np.hstack([list(graph.nodes())[x] for x in range(len(xs[0]))] * len(xs))
	xs = np.hstack(xs)
	df = pd.DataFrame({'time': time, 'mapping': mapping, 'probabilities': xs})
	plt.figure(figsize=(12,8))
	sns.tsplot(df, time='time', unit = "mapping",
	               condition='mapping', value='probabilities')
