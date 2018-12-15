# SandP500-index-price-prediction
Udacity DSND Capstone project- S&P Index price prediction.

Project Overview
To predict S&P 500 Index prices using historical data since 1871 and statistical models like Linear
Regression, LSTM(Long Short-term Memory) and fbProphet.

Data source-
Data source for this project is Quandl API (MULPL). I am using my personal key to download data from MULPL api which is for free. I have also created 3 csv with merged dataframe, imputed and interpolated data to remove missing values which get introduced due to different dates in time-series for quater and year variables.
SandP500_Index_df_imputed.csv
SandP500_Index_df_Interpolated.csv
SandP500_Index_Master.csv

How to use the notebook-
Download the jupyter notebook and run in Python 3 environment. Entire jupyter notebook runs on its own without any interaction required.
Notebook would if there are missing prerequisite libraries. There is no help file on libraries except for the one mentioned in the jupyter notebook. Notebook downloads data from the Qandl API every time from the beginning when notebook is run from the start.
