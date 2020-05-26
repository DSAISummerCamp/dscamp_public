##
# Importing modules and loading the data
# Note that statsmodels are used for logistic regression instead of sklearn
# The main reason - coefficient significant and testing is not available out of the box for sklearn
# There are minor differences in the model specification between logistic regression on statsmodels/sklearn
# See package documentation for more details
##
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as logit
from sklearn.preprocessing import LabelEncoder as encoder
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
data = pd.read_csv('/content/gdrive/My Drive/dscamp/dscamp_utils/Linear Models/face_data_inf.csv')
RANDOM_SEED = 123
