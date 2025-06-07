# Data analysis packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import dataset
df = pd.read_excel('News Title.xls')

# Show the first 5 rows the dataset
print(df.head(5))