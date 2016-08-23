import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB, MultinomialNB



class ModelEvaluationHelper(object):
	"""docstring for ModelEvaluationHelper"""
	def __init__(self, X, y, test_size, random_state, label_names):
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			X, y, test_size = test_size, random_state = random_state)
		self.label_names = label_names

	def set_hyperparam_grid(self, params):
		self.hyperparameters = params		

	def cross_val_grid_search(self, classifier, scoring, cv=5, iid=True, verbose=1):

		self.grid_search =  GridSearchCV(
			classifier, self.hyperparameters, cv=cv, scoring=scoring, iid=iid, verbose=verbose)
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
		print(classification_report(self.y_test, y_pred, target_names=self.label_names))
		print()

	def _plot_confusion_matrix(self, cm, cmap=plt.cm.Blues):
	    plt.imshow(cm, interpolation='nearest', cmap=cmap)
	    plt.title('Normalized confusion matrix')
	    plt.colorbar()
	    tick_marks = np.arange(len(self.label_names))
	    plt.xticks(tick_marks, self.label_names, rotation=45)
	    plt.yticks(tick_marks, self.label_names)
	    plt.tight_layout()
	    plt.ylabel('True label')
	    plt.xlabel('Predicted label')

	def confusion_matrix(self):
		# Compute confusion matrix
		y_pred = self.grid_search.predict(self.X_test)
		cm = confusion_matrix(self.y_test, y_pred)
		cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		np.set_printoptions(precision=2)

		plt.figure(figsize=(8,8))
		self._plot_confusion_matrix(cm_normalized)
