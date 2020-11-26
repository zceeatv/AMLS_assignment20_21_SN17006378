import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import os
import pandas as pd
from os.path import dirname, abspath

basedir = dirname(dirname(abspath(__file__)))
labels_filename = 'datasets/celeba/labels.csv'

column = [0, 2]
df = pd.read_csv(os.path.join(basedir, labels_filename), delimiter="\t", usecols=column)
gender_labels = df.to_numpy()
print(gender_labels)

