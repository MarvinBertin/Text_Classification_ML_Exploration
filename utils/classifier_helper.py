from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesClassifier


class Classifier(object):
	"""docstring for Classifier"""
	def __init__(self):
		pass

	def multinomial_SGD_clf(
		self, model, n_iter=5, shuffle=True, learning_rate='optimal', class_weight=None, average=False):
		if model == "linearSVM":
			loss = "hinge"
		elif model == "logisticRegression":
			loss = "log"
		elif model == "neuralNet":
			loss = "perceptron"
		else:
			loss = model #ie ‘modified_huber’, ‘squared_hinge’
		
		return SGDClassifier(
		loss=loss, n_iter=n_iter, shuffle=shuffle, learning_rate=learning_rate,
		class_weight=class_weight, average=average, n_jobs=-1)

	def multinomial_NB_clf(self, model):
		if model == "multinomialNB":
			return MultinomialNB(fit_prior=True)
		elif model == "bernoulliNB":
			return BernoulliNB(fit_prior=True)

	def multinomial_neighbors_clf(
		self, model, metric='euclidean', n_neighbors=5, weights='uniform'):
		if model == "NearestCentroid":
			return NearestCentroid(metric=metric)
		if model == "KNN":
			return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=-1)

	#[‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
	# "uniform", "distance"
	# “euclidean”, “manhattan”, “chebyshev”, “minkowski”, “wminkowski”, “seuclidean”, “mahalanobis”

	def multinomial_SVM_clf(self, model, class_weight="balanced", degree=3, shrinking=True, probability=False):
		if model == "linearSVM":
			kernel = "linear"
		elif model == "gaussianSVM":
			kernel = "rbf"
		elif model == "polynomialSVM":
			kernel = "poly"
		return SVC(kernel=kernel, class_weight=class_weight, degree=degree, shrinking=shrinking, probability=probability)

	def multinomial_Ensemble_clf(self, model, class_weight="balanced"):
		if model == "randomForest":
			return RandomForestClassifier(class_weight=class_weight, n_jobs=-1)
		elif model == "extraTrees":
			return ExtraTreesClassifier(class_weight=class_weight, n_jobs=-1)
		elif model == "adaBoost":
			return AdaBoostClassifier()

		
		