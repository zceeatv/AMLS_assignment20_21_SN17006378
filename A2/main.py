import os
import pandas as pd
from os.path import dirname, abspath

basedir = dirname(dirname(abspath(__file__)))
labels_filename = 'datasets/celeba/labels.csv'

column = [0, 3]
df = pd.read_csv(os.path.join(basedir, labels_filename), delimiter="\t", usecols=column)
smile_labels = df.to_numpy()
print(smile_labels)