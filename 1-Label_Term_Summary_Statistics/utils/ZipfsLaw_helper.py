import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ZipfsLawHelper(object):
    
    def __init__(self, doc_term_matrix):
        self.y = doc_term_matrix.sum(axis = 0).getA().squeeze()
        self.x = np.array(range(1,len(self.y)+1))

    def _powerLaw(self):
        """
        'When the frequency of an event varies as power of some attribute of that
        event the frequency is said to follow a power law.' (wikipedia)

        This is represented by the following equation, where c and alpha are
        constants:
        y = c . x ^ alpha

        Args
        --------
        y: array with frequency of events >0
        x: numpy array with attribute of events >0

        Output
        --------
        (c, alpha)

        c: the maximum frequency of any event
        alpha: defined by (Newman, 2005 for details):
            alpha = 1 + n * sum(ln( xi / xmin )) ^ -1
        """
        c = 0
        alpha = .0

        if len(self.y) and len(self.y)==len(self.x):
            c = max(self.y)
            xmin = float(min(self.x))
            alpha = 1 + len(self.x) * pow(sum(np.log(self.x/xmin)),-1)

        return (c, alpha)

    def _plotPowerLaws(self, c=[], alpha=[]):
        """
        Plots the relationship between x and y and a fitted power law on LogLog
        scale.

        Args
        --------
        y: array with frequency of events >0
        x: array with attribute of events >0
        c: array of cs for various power laws
        alpha: array of alphas for various power laws
        """
        plt.figure(figsize=(10, 8))
        plt.title("Zipfs law: Term Rank vs Term frequency on Log-Log Plot", fontsize=20)
        plt.loglog()
        plt.plot(self.x, sorted(self.y, reverse=True), 'r*', label = "corpus")
        plt.xlabel("Term Frequency")
        plt.ylabel("Term Rank")

        for _c, _alpha in zip(c,alpha):
            plt.plot( (1, max(self.x)),
                      (_c, _c * pow(max(self.x), _alpha)),
                      label='Zipfs law ~x^%.2f' % _alpha)
            plt.legend()
        plt.show()
        
    def plot_corpus_term_freq(self):
        plt.figure(figsize=(10, 8))
        plt.title("Full Vocabulary Term Frequencies", fontsize=24)
        plt.ylabel("Count")
        plt.xlabel("Term IDs")
        plt.plot(self.x, self.y)
        plt.show()
        
    def plot_Zipfs_Law(self):
        c, alpha = self._powerLaw()
        print('According to Zipfs law %.2f should be close to 1.00' % alpha)
        self._plotPowerLaws([c,c], [-1,-alpha])