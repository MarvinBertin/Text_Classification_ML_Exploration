import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataLoader(object):
    """docstring for data_loader"""
    def __init__(self):
        pass

    def load_data(self, fname):

        with open(fname) as f:
            content = f.readlines()

        print("Number of Rows: ", content[0])
        return content

    def _extract_data(self, line):
        return (line[0], line[1:])

    def get_data_frame(self, content, label_names):

        data = list(map(self._extract_data, content[1:]))
        label_idx = np.array(list(map(lambda line: int(line[0])-1, data)))
        labels = label_names[label_idx]
        sentences = list(map(lambda line: line[1].strip(), data))

        return pd.DataFrame({"Labels": labels, "Text": sentences})