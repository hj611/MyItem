import pandas as pd 
import numpy as np
import os 

dataOfBlack_2020_Q1 = []

dirs = os.listdir('E:/WJ/data_Q1_2018')
for i in dirs:
    dataOfBlack_2020_Q1.append(pd.read_csv('E:/WJ/data_Q1_2018/' + i))

for i in range(len(dataOfBlack_2020_Q1)):
    dataOfBlack_2020_Q1[i] = dataOfBlack_2020_Q1[i][dataOfBlack_2020_Q1[i]['model'] == 'ST4000DM000']

'''

for i in range(len(dataOfBlack_2020_Q1)):
    models = list(set(dataOfBlack_2020_Q1[i]['model'].values))
    print(models)
    print('**************************************************')
    for model in list(set(dataOfBlack_2020_Q1[i]['model'].values)):
        if(not model.startswith('ST')):
            models.remove(model)
    print(models)
    dataOfBlack_2020_Q1[i] = dataOfBlack_2020_Q1[i][dataOfBlack_2020_Q1[i]['model'].isin(models)]


'''

for i in range(len(dataOfBlack_2020_Q1)):
    dataOfBlack_2020_Q1[i].to_csv('E:/WJ/data_Q1_2018_b/' + dirs[i])
    