
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA, NMF, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import FeatureAgglomeration
from sklearn import manifold
from sklearn.preprocessing import scale
from sklearn.cross_validation import cross_val_score

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from sklearn.grid_search import GridSearchCV


class DimensionalityReducer(object):
	"""docstring for SignalDecomposer"""
	def __init__(self, X):
		self.X = X.toarray()

	def _explained_variance(self, reducer):
		if hasattr(reducer, "explained_variance_ratio_"):
			explained_variance = reducer.explained_variance_ratio_.sum()
			print("Explained variance of the signal decomposition step: {}%".format(int(explained_variance * 100)))

	def fit_reducer(self, model, n_components, n_iter = 5, kernel="rbf", gamma=None, alpha=.1, l1_ratio=.5):
		if model == "PCA":
			reducer = PCA(n_components=n_components)
		elif model == "TruncatedSVD":
			reducer = TruncatedSVD(n_components=n_components, n_iter=n_iter)
		elif model == "KernelPCA":
			reducer = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
		elif model == "NMF":
			reducer = NMF(n_components=n_components, alpha=alpha, l1_ratio=l1_ratio)
		elif model == "LDA":
			recuder = LinearDiscriminantAnalysis(n_components=n_components)
		elif model == "FactorAnalysis":
			reducer = FactorAnalysis(n_components=n_components)
		elif model == "GaussianRandom":
			reducer = GaussianRandomProjection(n_components=n_components)
		elif model == "SparseRandom":
			reducer = SparseRandomProjection(n_components=n_components)
		elif model == "FeatureAgglomeration":
			reducer = FeatureAgglomeration(n_clusters=n_componentss)

		reduced_X = reducer.fit_transform(self.X)
		self._explained_variance(reducer)
		return reduced_X

	def plot_TSNE_projection(self, X, y):
		tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
		X_tsne = tsne.fit_transform(X).T
		plt.figure(figsize=(10,8))
		plt.scatter(X_tsne[0], X_tsne[1], c=y, cmap=plt.cm.rainbow)
		plt.title("t-distributed Stochastic Neighbor Embedding (t-SNE)")
		plt.axis('tight')
		plt.legend()
		plt.show()

	def _compute_scores(self, reducer, n_components):

	    scores = []
	    for n in n_components:
	        reducer.n_components = n
	        scores.append(np.mean(cross_val_score(reducer, self.X)))
	    return scores

	def _run_CV(self, reducer, n_components):
		scores = self.compute_scores(reducer, n_components)
		component = n_components[np.argmax(scores)]
		return scores, component

	def _plot_scores(self, scores, component, n_components, reducer_name):
		print("best n_components by {} CV = {}".format(reducer_name, component))
		plt.plot(n_components, scale(scores), 'b', label=reducer_name +"scores")
		plt.axvline(component, color='b',
		            label='{} CV: {}'.format(reducer_name, component), linestyle='--')


	def plot_model_selection(self, n_components):
		plt.figure(figsize=(8, 6))

		for reducer_name in ["PCA", "Factor Analysis"]:
			if reducer_name == "PCA":
				reducer = PCA()
			else:
				reducer = FactorAnalysis()

			scores, component = self._run_CV(reducer, n_components)
			self._plot_scores(scores, component, n_components, reducer_name)

		plt.xlabel('nb of components')
		plt.ylabel('CV scores')
		plt.legend(loc='lower right')
		plt.title("Model selection with Probabilistic PCA and Factor Analysis ")
		plt.show()






				