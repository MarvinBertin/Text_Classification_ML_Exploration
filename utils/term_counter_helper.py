import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class TermCounter(object):
	"""docstring for TermFrequency"""
	def __init__(self,
		label_names,		
		lowercase,
		preprocessor,
		tokenizer,
		stop_words,
		ngram_range,
		analyzer,
		max_df,
		min_df,
		max_features,
		vocabulary):

		self.min_count = min_df
		self.label_names = label_names
		
	def vectorize_corpus(self, corpus):
		self.X = self.vectorizer.fit_transform(corpus)
		self.vocab = np.array(self.vectorizer.get_feature_names())

	def _bottom_n_filter(self, word_sorted, count2idx, n):
	    
	    bottom_n = []
	    bottom_n_frq = []
	    counter = 0
	    
	    for idx in word_sorted:
	        if count2idx[idx] >= self.min_count:
	            bottom_n.append(idx)
	            bottom_n_frq.append(count2idx[idx])
	            counter += 1
	        if counter == n:
	            break
	    return (bottom_n, bottom_n_frq)

	def _top_bottom_n_terms(self, word_count, n):

	    count2idx = dict(enumerate(word_count))
	    word_sorted = word_count.argsort()

	    top_n = word_sorted[-n:]
	    top_n_frq = [count2idx[idx] for idx in top_n]

	    bottom_n, bottom_n_frq = self._bottom_n_filter(word_sorted, count2idx, n)

	    return (list(zip(top_n, top_n_frq)), list(zip(bottom_n,bottom_n_frq)))

	def _term_freq_data_frame(self, term_freq):
	    terms = np.array([x[0] for x in term_freq])
	    freq = np.array([x[1] for x in term_freq])
	    
	    return pd.DataFrame({"Terms": self.vocab[terms],
	                         "Count": freq})

	def _plot_term_freq(self, data_frame, ax, palette):
	    ax = sns.barplot(y="Terms", x="Count", palette=palette, data=data_frame, ax=ax)
	    ax.set(xlabel = self.counter_name, ylabel='')

	def _run_term_freq_plot(self, word_count, N, title, figsize=(15, 10)):

	    top_n, bottom_n = self._top_bottom_n_terms(word_count, N)

	    f, axes = plt.subplots(1, 2, figsize=figsize, sharex=False)
	    f.suptitle(title, fontsize=25)

	    data_bottom = self._term_freq_data_frame(bottom_n)
	    self._plot_term_freq(data_bottom, axes[0], "Blues")

	    data_top = self._term_freq_data_frame(top_n)
	    self._plot_term_freq(data_top, axes[1], "Reds")

	def plot_term_freq_dist(self, N, per_label = False):

		if per_label:
			for idx, label in enumerate(self.label_names):
				word_count = self.X.getrow(idx).toarray().squeeze()
				self._run_term_freq_plot(
					word_count,
					N,
					title = "Top-N & Bottom-N {} (Label {})".format(self.counter_name, label),
					figsize=(15, 5))
		else:
			word_count = self.X.sum(axis = 0).getA().squeeze()
			self._run_term_freq_plot(
				word_count,
				N,
				title = "Top-N & Bottom-N {} (Overall)".format(self.counter_name),
				figsize=(15, 10))


class TermFrequency(TermCounter):

	def __init__(self,
		label_names,
		lowercase = True,
		preprocessor = None,
		tokenizer = None,
		stop_words = None,
		ngram_range = (1, 1),
		analyzer = 'word',
		max_df = 1.0,
		min_df = 1,
		max_features = None,
		vocabulary = None):

		super().__init__(
			label_names,
			lowercase,
			preprocessor,
			tokenizer,
			stop_words,
			ngram_range,
			analyzer,
			max_df,
			min_df,
			max_features,
			vocabulary)

		self.vectorizer = CountVectorizer(
			lowercase = lowercase,
			preprocessor = preprocessor,
			tokenizer = tokenizer,
			stop_words = stop_words,
			ngram_range = ngram_range,
			analyzer = analyzer,
			max_df = max_df,
			min_df = min_df,
			max_features = max_features,
			vocabulary = vocabulary)

		self.counter_name = "Term Frequency"


class TfIdf(TermCounter):

	def __init__(self,
		label_names,
		norm='l2',
		smooth_idf = True,
		sublinear_tf = False,
		lowercase = True,
		preprocessor = None,
		tokenizer = None,
		stop_words = None,
		ngram_range = (1, 1),
		analyzer = 'word',
		max_df = 1.0,
		min_df = 1,
		max_features = None,
		vocabulary = None):

		super().__init__(
			label_names,
			lowercase,
			preprocessor,
			tokenizer,
			stop_words,
			ngram_range,
			analyzer,
			max_df,
			min_df,
			max_features,
			vocabulary)

		self.vectorizer = TfidfVectorizer(
			norm='l2',
			smooth_idf = True,
			sublinear_tf = False,
			lowercase = lowercase,
			preprocessor = preprocessor,
			tokenizer = tokenizer,
			stop_words = stop_words,
			ngram_range = ngram_range,
			analyzer = analyzer,
			max_df = max_df,
			min_df = min_df,
			max_features = max_features,
			vocabulary = vocabulary)

		self.counter_name = "Tf-Idf Weight"
		self.min_count = 0.0001

