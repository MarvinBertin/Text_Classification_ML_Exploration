import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class DataFrameHelper(object):
	"""docstring for PlotHelper"""
	def __init__(self, dataframe, label_names):
		self.df = dataframe
		self.raw_text = self.df.Text.get_values()
		self.raw_labels = self.df.Labels.get_values()
		self.y = LabelEncoder().fit_transform(self.raw_labels)
		self.label_names = label_names
		self.groupby_label_df = self.groupby_label()
		

	def _data_transform(self, data):
	    return pd.Series({"Label_freq": data["Labels"].count(),
	                      "Text_concat": " ".join(data["Text"])
	                     })

	def groupby_label(self):
		self.document_df = self.df.groupby("Labels").apply(self._data_transform)
		self.corpus_matrix = self.document_df["Text_concat"].as_matrix()
		return self.document_df

	def plot_label_distribution(self):
		plt.figure(figsize=(10, 8))
		sns.countplot(x="Labels", data=self.df, palette="Paired")
		plt.title("Class Label Distribution", fontsize=24)
		plt.show()

	def global_corpus_summary(self, doc_term_matrix, min_df):
		label_num, vocab_size = doc_term_matrix.shape
		print("Data contains {} topic labels with {} unique words (minimum frequency of {})"
		      .format(label_num, vocab_size, min_df))

	def plot_terms_per_label(self, doc_term_matrix, unique = True):
	    label_num, _ = doc_term_matrix.shape
	    
	    if unique:
	        name = "Unique"
	        terms = np.array([doc_term_matrix.getrow(row).nnz for row in range(label_num)])
	    else:
	        name = "Total"
	        terms = doc_term_matrix.sum(axis = 1).getA().squeeze()
	    
	    df_terms = pd.DataFrame({"Labels": self.label_names,
	                             "{}_Terms".format(name): terms})
	    
	    plt.figure(figsize=(10, 8))
	    sns.barplot(x="Labels", y="{}_Terms".format(name), data=df_terms, palette="Paired")
	    plt.title("Number of {} Terms Per Label".format(name), fontsize=24)
	    plt.ylabel("Count")
	    plt.show()
		