# encoding: utf-8
import numpy as np
import pandas as pd

df = pd.read_table('/path/to/your/long-merge-kmer-file',sep='\s+', low_memory=False)

print(df.head())

df.columns = ['kmer','count','Sample']
print(df.head())
#Converted into strain number as column data
da = df.pivot_table(index=['Sample'],columns=['kmer'],values=['count'])

#Fill NA area
da.fillna(0,inplace=True) 

da.to_csv("/path/to/csv",index=True,sep=',')


