import pandas as pd
import os

file_path = '/workspace/China_Journal/raw_data/shanghai/data_6.1~6.15.xlsx'
try:
    df = pd.read_excel(file_path, nrows=5)
    print("Columns:", df.columns.tolist())
    print(df.head())
except Exception as e:
    print(e)
