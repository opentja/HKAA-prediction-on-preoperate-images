import pandas as pd
import os
import glob
def deleteInvalidAnnotationsFromTempCSV(csvPath):
    df = pd.read_csv(csvPath)
    indexNames = df[df['region_count'] != 6].index
    # Delete these row indexes from dataFrame
    df.drop(indexNames, inplace=True)
    return df

def getValidAnnotationIdFromTempCSV(csvPath):
    imagesIdList = []
    df = pd.read_csv(csvPath)
    indexNames = df[df['region_count'] == 20].index
    for index in range(len(indexNames)):
        idxName = indexNames[index]
        imageId = df.iloc[idxName, 0]
        if imageId not in imagesIdList:
            imagesIdList.append(imageId)
    return imagesIdList

def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def getUnionOfImagesId(imagesPath, imagesIdList):
    imagesNameList = set(os.listdir(imagesPath))
    unionId = intersection(imagesNameList, imagesIdList)
    return unionId

def deleteInvalidImages(imagesPath, unionId):
    for jpgPath in glob.glob(imagesPath+'*.jpg'):
        slashIndex = jpgPath.rfind('/')
        imageName = jpgPath[slashIndex+1:]
        if imageName not in unionId:
            os.remove(jpgPath)

def clearDataFrame(df, unionId, flag):
    df = df[df['#filename'].isin(unionId)]
    df.to_csv('randomSelect50/ankleToHip'+flag+'/clean.csv', index=False)





csvPath = 'allOAIdata/temporary.csv'
df = deleteInvalidAnnotationsFromTempCSV(csvPath)
df.to_csv('allOAIdata/clean.csv', index=False)
