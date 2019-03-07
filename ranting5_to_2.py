# -*- coding: utf-8 -*-
"""
Created on 07/03/2019
@author: Haipeng 
"""
import pandas as pd 
import numpy as np

buffer_df = pd.read_csv('SleepQuality_After_Cleaning.csv')

'''将5个指标聚合为三个:
    1-2: 0 (Bad)
    3:   1 (Mean)
    4-5: 2 (Good)
'''
buffer_df.ms204a[buffer_df['ms204a'] <= 3] = 0
buffer_df.ms204a[buffer_df['ms204a'] >  3] = 1


buffer_df.ms204b[buffer_df['ms204b'] <= 3] = 0
buffer_df.ms204b[buffer_df['ms204b'] >  3] = 1


buffer_df.ms204c[buffer_df['ms204c'] <= 3] = 0
buffer_df.ms204c[buffer_df['ms204c'] >  3] = 1

buffer_df.to_csv('SleepQuality_DNN_Classification_binary.csv',index=False,header=True)