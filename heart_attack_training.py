#Maths
import numpy as np
import pandas as pd
import math
import time

#Graphs and Plots
import sweetviz as vs
import PIL as Image
import matplotlib.pyplot as plt
import seaborn as sns

#Data Processing
from sklearn import pipeline
from sklearn import preprocessing #OrdinalEncoder and OHE
from sklearn import impute
from sklearn import compose
from sklearn import model_selection #train_test_split
from sklearn import metrics #accuracy score, balanced_accuracy_score, confusion_matrix

#Within Libraries
import heart_attack_pipeline as ha_pl


# STEP 1
# Read Data
df = pd.read_csv("data/heart.csv")

# STEP 2
# Call Pipelines for Training the Machine Learning Model
ha_pl.training_pipeline(df,save_model=True)

