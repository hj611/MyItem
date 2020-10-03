import pandas as pd
import os 
dataOfBlack_2020_Q1 = []
dirs_Q1 = os.listdir('E:/WJ/data_Q1_2018_b')
dirs_Q2 = os.listdir('E:/WJ/data_Q2_2018_b')
dirs_Q3 = os.listdir('E:/WJ/data_Q3_2018_b')
for i in dirs_Q1:
    dataOfBlack_2020_Q1.append(pd.read_csv('E:/WJ/data_Q1_2018_b/' + i))

for i in dirs_Q2:
    dataOfBlack_2020_Q1.apppip end(pd.read_csv('E:/WJ/data_Q2_2018_b/' + i))

for i in dirs_Q3:
    dataOfBlack_2020_Q1.append(pd.read_csv('E:/WJ/data_Q3_2018_b/' + i))

Sample = pd.concat(dataOfBlack_2020_Q1, ignore_index=True)
data = Sample.sort_values(by=['serial_number', 'date'])


drop_list = []
for i in [col for col in data.columns if col not in ['date','serial_number', 'model', 'capacity_bytes', 'failure']]:
    if(data[i].nunique() == 1 & data[i].isna().sum() == 0):
        drop_list.append(i)

fea = list(set(data.columns) - set(drop_list))

fea.remove('Unnamed: 0')
fea.sort()

data = data[fea]

data.fillna(method = 'pad', inplace = True)

select_fea = ['capacity_bytes', 'date', 'failure','model', 'serial_number', 'smart_190_raw', 
'smart_193_raw', 'smart_194_raw', 'smart_1_raw', 'smart_241_raw', 'smart_7_raw']



data = data[select_fea]

final_store = pd.HDFStore('E:/WJ/final_BlackBlaze_Q1_Q2_Q3.h5')
final_store.append('final_BlackBlaze_Q1_Q2_Q3', data)
final_store.close()