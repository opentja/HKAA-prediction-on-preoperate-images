import pandas as pd
import shutil
import numpy as np
import os


def readCSV():
    df25= pd.read_csv('./smallerThan25.csv')
    df20 = pd.read_csv('./smallerThan20.csv')
    df15 = pd.read_csv('./smallerThan15.csv')
    df10 = pd.read_csv('./smallerThan10.csv')
    df05 = pd.read_csv('./smallerThan5.csv')

    # extract bigger than 5
    # newCSV = df10[(df10['leftLegAngle'] < -5) | (df10['leftLegAngle'] > 5) | (df10['rightLegAngle'] < -5) | (
    #         df10['rightLegAngle'] > 5)]
    # imageNameList = newCSV['imageName'].tolist()
    # df5 = df10[~df10['imageName'].isin(imageNameList)]
    # df5.to_csv('./smallerThan5.csv',index=False)

    nameList20 = df20['imageName'].tolist()
    nameList15 = df15['imageName'].tolist()
    nameList10 = df10['imageName'].tolist()
    nameList05 = df05['imageName'].tolist()

    df2025 = df25[~df25['imageName'].isin(nameList20)]['imageName'].tolist()
    df1520 = df20[~df20['imageName'].isin(nameList15)]['imageName'].tolist()
    df1015 = df15[~df15['imageName'].isin(nameList10)]['imageName'].tolist()
    df0510 = df10[~df10['imageName'].isin(nameList05)]['imageName'].tolist()
    df0005 = nameList05
    return df2025, df1520, df1015, df0510, df0005


def randomCopy(nameList, path, flag):
    newPath = '/infodev1/phi-data/shi/kneeX-ray/preoperate/ankleToHip' + flag + '/'
    if not os.path.exists(newPath):
        os.makedirs(newPath)
    end = len(nameList)
    indexList = np.random.randint(0, end, 50)
    for idx in indexList:
        shutil.copy(path + nameList[idx], newPath)


preoperate = '/infodev1/phi-data/shi/kneeX-ray/preoperate/ankleToHip/'

df2025, df1520, df1015, df0510, df0005 = readCSV()

randomCopy(df2025, preoperate, '2025')
randomCopy(df1520, preoperate, '1520')
randomCopy(df1015, preoperate, '1015')
randomCopy(df0510, preoperate, '0510')
randomCopy(df0005, preoperate, '0005')
