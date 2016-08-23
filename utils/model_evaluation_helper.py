import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.utils.extmath import density


class ModelEvaluationHelper(object):
	"""docstring for ModelEvaluationHelper"""
	def __init__(self, X, y, test_size, random_state, label_names, feature_names = None):
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			X, y, test_size = test_size, random_state = random_state)
		self.label_names = label_names
		self.feature_names = np.asarray(feature_names)

		print("Number of observations in Train: %d" % len(self.y_train))
		print("Number of observations in Test : %d" % len(self.y_test))

	def set_hyperparam_grid(self, params):
		self.hyperparameters = params		

	def cross_val_grid_search(self, classifier, scoring, cv=5, iid=True, print_top_10=True, verbose=1):

		self.grid_search =  GridSearchCV(
			classifier, self.hyperparameters, cv=cv, scoring=scoring, iid=iid, verbose=verbose)
		
		self.print_header("Cross-Validation & Grid Search")
		self.grid_search.fit(self.X_train, self.y_train)
		self._print_density(print_top_10)
		self._print_results()

	def _print_density(self, print_top_10):
		clf = self.grid_search.best_estimator_
		try:
			if hasattr(clf, 'coef_'):
				self.print_header("Matrix Sparsity")
				print("dimensionality: %d" % clf.coef_.shape[1])
				print("density: %f" % density(clf.coef_))

				if print_top_10 and self.feature_names is not None:
					self.print_header("Feature Importants")
					print("top 10 keywords per class:")
					for i, label in enumerate(self.label_names):
						top10 = np.argsort(clf.coef_[i])[-10:]
						print(self.trim("%s: %s"
							% (label, ", ".join(self.feature_names[top10]))))
					print()
		except:
			self.print_header("Matrix Sparsity")
			print("matrix coefficents are only available when using a linear kernel")

	def _print_results(self):
		self.print_header("Best Model Results")
		print("Best score: %0.3f" % self.grid_search.best_score_)
		print("Best parameters set:")
		best_parameters = self.grid_search.best_estimator_.get_params()
		for param_name in sorted(self.hyperparameters.keys()):
		    print("\t%s: %r" % (param_name, best_parameters[param_name]))

		self.print_header("Grid Search")
		print("Grid scores on training set:")
		print()
		for params, mean_score, scores in self.grid_search.grid_scores_:
		    print("%0.3f (+/-%0.03f) for %r"
		          % (mean_score, scores.std() * 2, params))
		print()

		self.print_header("Detailed classification report")
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

	def trim(self, s):
	    """Trim string to fit on terminal (assuming 80-column display)"""
	    return s if len(s) <= 80 else s[:77] + "..."

	def print_header(self, header):
		print("\n" + "=" * 80)
		print("* " + header + "\n")
