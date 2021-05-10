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
    df.to_csv('/infodev1/phi-data/shi/kneeX-ray/randomSelect50/ankleToHip'+flag+'/clean.csv', index=False)




# csvPath='/infodev1/phi-data/shi/kneeX-ray/randomSelect50/ankleToHip0005/annotation.csv'
# imagesPath = '/infodev1/phi-data/shi/kneeX-ray/randomSelect50/ankleToHip0005/'
# tempDf = deleteInvalidAnnotationsFromTempCSV(csvPath)
# imagesIdList = getValidAnnotationIdFromTempCSV(csvPath)
# unionId = getUnionOfImagesId(imagesPath, imagesIdList=imagesIdList)
# deleteInvalidImages(imagesPath, unionId)
# clearDataFrame(tempDf, unionId, '0005')
#
# csvPath='/infodev1/phi-data/shi/kneeX-ray/randomSelect50/ankleToHip0510/annotation.csv'
# imagesPath = '/infodev1/phi-data/shi/kneeX-ray/randomSelect50/ankleToHip0510/'
# tempDf = deleteInvalidAnnotationsFromTempCSV(csvPath)
# imagesIdList = getValidAnnotationIdFromTempCSV(csvPath)
# unionId = getUnionOfImagesId(imagesPath, imagesIdList=imagesIdList)
# deleteInvalidImages(imagesPath, unionId)
# clearDataFrame(tempDf, unionId, '0510')
#
# csvPath='/infodev1/phi-data/shi/kneeX-ray/randomSelect50/ankleToHip1015/annotation.csv'
# imagesPath = '/infodev1/phi-data/shi/kneeX-ray/randomSelect50/ankleToHip1015/'
# tempDf = deleteInvalidAnnotationsFromTempCSV(csvPath)
# imagesIdList = getValidAnnotationIdFromTempCSV(csvPath)
# unionId = getUnionOfImagesId(imagesPath, imagesIdList=imagesIdList)
# deleteInvalidImages(imagesPath, unionId)
# clearDataFrame(tempDf, unionId,'1015')


# csvPath='/infodev1/phi-data/shi/kneeX-ray/randomSelect50/ankleToHip1520/annotation.csv'
# imagesPath = '/infodev1/phi-data/shi/kneeX-ray/randomSelect50/ankleToHip1520/'
# tempDf = deleteInvalidAnnotationsFromTempCSV(csvPath)
# imagesIdList = getValidAnnotationIdFromTempCSV(csvPath)
# unionId = getUnionOfImagesId(imagesPath, imagesIdList=imagesIdList)
# deleteInvalidImages(imagesPath, unionId)
# clearDataFrame(tempDf, unionId,'1520')



# csvPath='/infodev1/phi-data/shi/kneeX-ray/randomSelect50/ankleToHip2025/annotation.csv'
# imagesPath = '/infodev1/phi-data/shi/kneeX-ray/randomSelect50/ankleToHip2025/'
# tempDf = deleteInvalidAnnotationsFromTempCSV(csvPath)
# imagesIdList = getValidAnnotationIdFromTempCSV(csvPath)
# unionId = getUnionOfImagesId(imagesPath, imagesIdList=imagesIdList)
# deleteInvalidImages(imagesPath, unionId)
# clearDataFrame(tempDf, unionId,'2025')
csvPath = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/temporary.csv'
df = deleteInvalidAnnotationsFromTempCSV(csvPath)
df.to_csv('/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/clean.csv', index=False)

