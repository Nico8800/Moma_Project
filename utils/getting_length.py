import pandas as pd
import json
import csv

train_df=pd.read_csv('/home/nicolasg/code/annotations/train1.csv')
dic={'trim_vid_id':[],
     'length':[]}

for i in range(len(train_df)):
    if train_df['trim_video_id'][i] not in dic['trim_vid_id']:
        dic['trim_vid_id'].append(train_df['trim_video_id'][i])
        dic['length'].append(train_df['length'][i])


with open('/home/nicolasg/code/annotations/lenghts.csv', 'w') as convert_file:
     convert_file.write(json.dumps(dic))