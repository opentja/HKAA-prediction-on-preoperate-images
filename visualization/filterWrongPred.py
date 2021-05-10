import pandas as pd
import os


df = pd.read_csv('./final.csv')
path1 = ''
path2 = ''
wrongPredList = os.listdir(path1)+os.listdir(path2)


newdf= df[~df['imageName'].isin(wrongPredList)]
newdf.to_csv('wrongPredFilted.csv',index=False)
