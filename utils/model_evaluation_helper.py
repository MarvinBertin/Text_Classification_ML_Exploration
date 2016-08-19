from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB, MultinomialNB


class ModelEvaluationHelper(object):
	"""docstring for ModelEvaluationHelper"""
	def __init__(self, X, y, test_size, random_state):
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			X, y, test_size = test_size, random_state = random_state)

	def set_hyperparam_grid(self, params):
		self.hyperparameters = params

	def _set_classifier(self, classifier):
		if classifier == "linearSVM":
			self.clf = SGDClassifier(loss="hinge", shuffle=True, n_jobs=-1)
		elif classifier == "logisticRegression":
			self.clf = SGDClassifier(loss="log", shuffle=True, n_jobs=-1)
		elif classifier == "neuralNet":
			self.clf = SGDClassifier(loss="perceptron", shuffle=True, n_jobs=-1)
		elif classifier == "multinomialNB":
			self.clf = MultinomialNB(fit_prior=True)
		elif classifier == "bernoulliNB":
			self.clf = BernoulliNB(fit_prior=True)
		else:
			self.clf = SGDClassifier(loss=classifier, shuffle=True, n_jobs=-1) #ie ‘modified_huber’, ‘squared_hinge’
		

	def cross_val_grid_search(self, classifier, scoring, cv=5, iid=True, verbose=1):

		self._set_classifier(classifier)
		self.grid_search =  GridSearchCV(
			self.clf, self.hyperparameters, cv=cv, scoring=scoring, iid=iid, verbose=verbose)
		self.grid_search.fit(self.X_train, self.y_train)
		self._print_results()

	def _print_results(self):
		print("Best score: %0.3f" % self.grid_search.best_score_)
		print("Best parameters set:")
		best_parameters = self.grid_search.best_estimator_.get_params()
		for param_name in sorted(self.hyperparameters.keys()):
		    print("\t%s: %r" % (param_name, best_parameters[param_name]))

		print()
		print("Grid scores on training set:")
		print()
		for params, mean_score, scores in self.grid_search.grid_scores_:
		    print("%0.3f (+/-%0.03f) for %r"
		          % (mean_score, scores.std() * 2, params))
		print()

		print("Detailed classification report:")
		print()
		print("The model is trained on the full train set with cross-validation.")
		print("The scores are computed on the full test set.")
		print()
		y_pred = self.grid_search.predict(self.X_test)
		print(classification_report(self.y_test, y_pred))
		print()