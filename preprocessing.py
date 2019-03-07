# -*- coding: utf-8 -*-
"""
Created on 07/03/2019
@author: Haipeng 
"""
import pandas as pd 
import numpy as np

# read in the sleep dataset
df = pd.read_csv('shhs2-dataset-0.13.0.csv', encoding= "ISO-8859-1")

"""
Feature selecting:

avg23bpd_s2: Average Diastolic blood pressure (BP)
avg23bps_s2: Average Systolic blood pressure (BP)
ai_all: Arousal Index
rdi0p： Overall Respiratory Disturbance Index (RDI) all oxygen desaturations -- The prevalence of obstructive sleep apnea (OSA) depends on the definition of the respiratory disturbance index (RDI) or apnea–hypopnea index (AHI) criteria，（Obstructive Sleep Apnea in Adults: Epidemiology and Variants）
nsupinep: Percent Time non-supine
pctlt75: Percent of sleep time with less than 75% oxygen saturation (SaO2)
pctlt80: Percent of sleep time with less than 80% oxygen saturation (SaO2)
pctlt85: Percent of sleep time with less than 85% oxygen saturation (SaO2)
pctlt90: Percent of sleep time with less than 90% oxygen saturation (SaO2), Ratio of the number of minutes with oxygen saturation (SaO2) under 90% to the total sleep time expressed in hours.
slp_eff: Percentage of time in bed that was spent sleeping, or the ratio of total sleep time to total time in bed, expressed as a percentage.
slp_lat: Time from lights out time to beginning of sleep, rounded to nearest minute
slpprdp: Sleep Time
slptime: Total Sleep Time
supinep: Percent Time supine
times34p: Percent Time in Stage 3/4
timest1p: Percent Time in Stage 1
timest2p: Percent Time in Stage 2
waso: Total amount of time spent awake after going to sleep
ms204a: Morning Survey (Sleep Heart Health Study Visit Two (SHHS2)), Quality of sleep light/deep, Rate the actual quality of your sleep last night (Do not compare to usual sleep quality). My sleep last night was (circle a number for each): a. [5 point Likert scale from "Light" to "Dark"]
ms204b: Morning Survey (Sleep Heart Health Study Visit Two (SHHS2)): Quality of sleep: short/long
ms204c: Morning Survey (Sleep Heart Health Study Visit Two (SHHS2)): Quality of sleep: restless/restful

"""
buffer_df = df.loc[:,['avg23bpd_s2','avg23bps_s2','ai_all','rdi0p', 'nsupinep', 'pctlt75', 'pctlt80', 'pctlt85', 'pctlt90',
                     'slp_eff', 'slp_lat', 'slpprdp', 'slptime', 'supinep', 'times34p', 'timest1p', 'timest2p', 'waso',
                     'ms204a', 'ms204b', 'ms204c']]

# Begin data cleaning
# Beacause it has too much null values, so we drop the "slptime"
buffer_df = buffer_df.drop(['slptime'], axis = 1)
buffer_df = buffer_df.dropna(
            axis = 0,
            how = 'all')
# We choose rows who have at least 16 non-empty values
buffer_df = buffer_df.dropna(axis=0, thresh=16)
# Calculate the average of each column
# buffer_df.mean()
buffer_df['avg23bpd_s2'] = buffer_df['avg23bpd_s2'].fillna(value=70.677064)
buffer_df['avg23bps_s2'] = buffer_df['avg23bps_s2'].fillna(value=127.766135)
buffer_df['ai_all'] = buffer_df['ai_all'].fillna(value=18.372544)
buffer_df['rdi0p'] = buffer_df['rdi0p'].fillna(value=27.944731)
buffer_df['nsupinep'] = buffer_df['nsupinep'].fillna(value=64.623896)
buffer_df['pctlt75'] = buffer_df['pctlt75'].fillna(value=0.046970)
buffer_df['pctlt80'] = buffer_df['pctlt80'].fillna(value=0.156439)
buffer_df['pctlt85'] = buffer_df['pctlt85'].fillna(value=0.635985)
buffer_df['pctlt90'] = buffer_df['pctlt90'].fillna(value=4.264773)
buffer_df['slp_eff'] = buffer_df['slp_eff'].fillna(value=79.182438)
buffer_df['slp_lat'] = buffer_df['slp_lat'].fillna(value=25.909047)
buffer_df['slpprdp'] = buffer_df['slpprdp'].fillna(value=374.065481)
buffer_df['supinep'] = buffer_df['supinep'].fillna(value=35.378943)
buffer_df['times34p'] = buffer_df['times34p'].fillna(value=15.948032)
buffer_df['timest1p'] = buffer_df['timest1p'].fillna(value=5.753917)
buffer_df['timest2p'] = buffer_df['timest2p'].fillna(value=57.712648)
buffer_df['waso'] = buffer_df['waso'].fillna(value=80.084671)
# For labels, we drop those null values
buffer_df.dropna(axis=0, how='any',inplace=True)

buffer_df.to_csv('SleepQuality_After_Cleaning.csv',index=False,header=True)







