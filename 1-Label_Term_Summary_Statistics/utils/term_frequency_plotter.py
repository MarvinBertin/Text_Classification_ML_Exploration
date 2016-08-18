import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

class TermFrequency(object):
	"""docstring for TermFrequency"""
	def __init__(self,
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
		
	def vectorize_corpus(self, corpus):
		self.X = self.vectorizer.fit_transform(corpus)

	def _top_bottom_n_terms(self, word_count, n):
    
		top_n_frq = sorted(word_count)[-n:]
		bottom_n_frq = sorted(word_count)[:n]

		top_n = word_count.argsort()[-n:]
		bottom_n = word_count.argsort()[:n]

		return (list(zip(top_n, top_n_frq)), list(zip(bottom_n,bottom_n_frq)))

	def _term_freq_data_frame(self, term_freq, vocab):
	    terms = np.array([x[0] for x in term_freq])
	    freq = np.array([x[1] for x in term_freq])
	    
	    return pd.DataFrame({"Terms": vocab[terms],
	                         "Count": freq})

	def _plot_term_freq(self, data_frame, ax, palette):
	    ax = sns.barplot(y="Terms", x="Count", palette=palette, data=data_frame, ax=ax)
	    ax.set(xlabel='Term Frequency', ylabel='')

	def run_term_freq_plot(self, word_count, N, title, figsize=(15, 10)):

	    (top_n, bottom_n) = self._top_bottom_n_terms(word_count, N)
	    vocab = np.array(self.vectorizer.get_feature_names())

	    f, axes = plt.subplots(1, 2, figsize=figsize, sharex=False)
	    f.suptitle(title, fontsize=25)

	    data_bottom = self._term_freq_data_frame(bottom_n, vocab)
	    self._plot_term_freq(data_bottom, axes[0], "Blues")

	    data_top = self._term_freq_data_frame(top_n, vocab)
	    self._plot_term_freq(data_top, axes[1], "Reds")


