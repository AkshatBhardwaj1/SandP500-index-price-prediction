# SandP500-index-price-prediction
Udacity DSND Capstone project- S&P Index price prediction.

# Project Overview
To predict S&P 500 Index prices using historical data since 1871 and statistical models like Linear
Regression, LSTM(Long Short-term Memory) and fbProphet.

# Data source-
Data source for this project is Quandl API (MULPL). I am using my personal key to download data from MULPL api which is for free. I have also created 3 csv with merged dataframe, imputed and interpolated data to remove missing values which get introduced due to different dates in time-series for quater and year variables.
SandP500_Index_df_imputed.csv
SandP500_Index_df_Interpolated.csv
SandP500_Index_Master.csv

# How to use the notebook-
Download the jupyter notebook and run in Python 3 environment. Entire jupyter notebook runs on its own without any interaction required.
Notebook would if there are missing prerequisite libraries. There is no help file on libraries except for the one mentioned in the jupyter notebook and here. Notebook downloads data from the Qandl API every time from the beginning when notebook is run from the start.

# Proposal and report-
Proposal.pdf has proposal for this project while Capstone Project Report.pdf has the project report with results and conclusion.

# LSTM- 
This project is based on LSTM using keras and fbProphet models.

List of libraries used-
import quandl
import pandas as pd
import numpy as np
import datetime as dt
import pandas_profiling
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
#import libraries here; add more as necessary
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import defaultdict
#Import supplementary visualization code visuals.py
#import visuals as vs
from numpy import concatenate
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# LSTM

#magic word for producing visualizations in notebook.allow plots to appear directly in the notebook
%matplotlib inline
from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import LSTM, CuDNNLSTM , BatchNormalization
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
from numpy import newaxis

# fbProphet libraries
from fbprophet import Prophet
#plt.style.available
plt.style.use("seaborn-whitegrid")
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import statsmodels.api as sm
from scipy import stats
