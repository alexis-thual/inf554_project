import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
from slugify import slugify
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.filters.filtertools import convolution_filter
from pandas.core.nanops import nanmean as pd_nanmean

data_ass = pd.read_csv('new_data/domicile.txt',sep='\t',parse_dates=[0],index_col=0)
data_ass.dropna(inplace=True)
