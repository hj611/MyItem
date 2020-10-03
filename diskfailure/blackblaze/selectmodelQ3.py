import pandas as pd 
import numpy as np
import os 

dataOfBlack_2020_Q3 = []

dirs = os.listdir('E:/WJ/data_Q3_2018')
for i in dirs:
    dataOfBlack_2020_Q3.append(pd.read_csv('E:/WJ/data_Q3_2018/' + i))

for i in range(len(dataOfBlack_2020_Q3)):
    dataOfBlack_2020_Q3[i] = dataOfBlack_2020_Q3[i][dataOfBlack_2020_Q3[i]['model'] == 'ST4000DM000']

'''
for i in range(len(dataOfBlack_2020_Q3)):
    models = list(set(dataOfBlack_2020_Q3[i]['model'].values))
    print(models)
    print('**************************************************')
    for model in list(set(dataOfBlack_2020_Q3[i]['model'].values)):
        if(not model.startswith('ST')):
            models.remove(model)
    print(models)
    dataOfBlack_2020_Q3[i] = dataOfBlack_2020_Q3[i][dataOfBlack_2020_Q3[i]['model'].isin(models)]
'''
for i in range(len(dataOfBlack_2020_Q3)):
    dataOfBlack_2020_Q3[i].to_csv('E:/WJ/data_Q3_2018_b/' + dirs[i])
    