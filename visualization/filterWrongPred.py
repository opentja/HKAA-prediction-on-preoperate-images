import pandas as pd
import os


df = pd.read_csv('./final.csv')
path1 = '/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/biggerThan20/wrongPred/'
path2 = '/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/biggerThan25/wrongPred/'
wrongPredList = os.listdir(path1)+os.listdir(path2)


newdf= df[~df['imageName'].isin(wrongPredList)]
newdf.to_csv('wrongPredFilted.csv',index=False)
