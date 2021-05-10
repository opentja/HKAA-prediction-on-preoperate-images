import os
import pandas as pd
import glob
import random
import shutil
import math


def extractPatientID(imagesPath):
    idList = []
    for jpgPath in glob.glob(imagesPath + '*.jpg'):
        # print(jpgPath )
        slashIdx = jpgPath.rfind('/')
        jpgName = jpgPath[slashIdx + 1:]
        
        # print(jpgName)
        patientID = jpgName.replace('.jpg', '')
        # print(patientID)
        if patientID not in idList:
            idList.append(patientID)
    random.shuffle(idList)
    return idList


def devideTrainAndTest(idList):
    oneFifth = int(math.floor(len(idList) / 5))
    tmpTrainIdList = idList[0:4 * oneFifth]
    testIdList = idList[4 * oneFifth:]
    return tmpTrainIdList, testIdList


def fiveCrossValidation(tmpTrainIdList, imagesPath, flag):
    validImageNameList = []
    trainImageNameList = []
    oneFifth = int(math.floor(len(tmpTrainIdList) / 5))

    index = imagesPath.find('/data/')
    replacedString = imagesPath[index + 5:]
    crossValidationPath = imagesPath.replace(replacedString, '/crossValidation/')
    df = pd.read_csv(imagesPath.replace('/images/', '/annotation.csv'))
    for i in range(0, 5, 1):
        valDf = df
        trainDf = df
        validationIdList = []
        trainIdList = []
        validImageNameList = []
        trainImageNameList = []
        if i != 4:
            tmpList = tmpTrainIdList[i * oneFifth:(i + 1) * oneFifth]
        else:
            tmpList = tmpTrainIdList[i * oneFifth:]
        validationIdList = tmpList
        trainIdList = list(set(tmpTrainIdList) - set(validationIdList))

        foldPath = crossValidationPath + flag + '/fold' + str(i) + '/'
        if not os.path.exists(foldPath):
            os.makedirs(foldPath)
        foldTrainImagePath = foldPath + 'train/images/'
        foldValidImagePath = foldPath + 'valid/images/'
        if not os.path.exists(foldTrainImagePath):
            os.makedirs(foldTrainImagePath)
        if not os.path.exists(foldValidImagePath):
            os.makedirs(foldValidImagePath)
        for id in validationIdList:
            if flag == 'hip':
                hipPath = imagesPath + id + 'hip.jpg'
                shutil.copy(hipPath, foldValidImagePath)
                hipImageName = id + 'hip.jpg'
                validImageNameList.append(hipImageName)
            if flag == 'knee':
                kneePath = imagesPath + id + 'knee.jpg'
                shutil.copy(kneePath, foldValidImagePath)
                kneeImageName = id + 'knee.jpg'
                validImageNameList.append(kneeImageName)

            if flag == 'ankle':
                anklePath = imagesPath + id + 'ankle.jpg'
                shutil.copy(anklePath, foldValidImagePath)
                ankleImageName = id + 'ankle.jpg'
                validImageNameList.append(ankleImageName)

        newValDf = valDf[valDf.imgName.isin(validImageNameList)]

        for id in trainIdList:
            if flag == 'hip':
                hipPath = imagesPath + id + 'hip.jpg'
                shutil.copy(hipPath, foldTrainImagePath)
                hipImageName = id + 'hip.jpg'
                trainImageNameList.append(hipImageName)
            if flag == 'knee':
                kneePath = imagesPath + id + 'knee.jpg'
                shutil.copy(kneePath, foldTrainImagePath)
                kneeImageName = id + 'knee.jpg'
                trainImageNameList.append(kneeImageName)
            if flag == 'ankle':
                anklePath = imagesPath + id + 'ankle.jpg'
                shutil.copy(anklePath, foldTrainImagePath)
                ankleImageName = id + 'ankle.jpg'
                trainImageNameList.append(ankleImageName)
        newTrainDf = trainDf[trainDf.imgName.isin(trainImageNameList)]

        newValDf.to_csv(foldValidImagePath.replace('/images/', '/annotation.csv'), index=False)
        newTrainDf.to_csv(foldTrainImagePath.replace('/images/', '/annotation.csv'), index=False)


def testData(testIdList, imagesPath, flag):
    testImageNameList = []
    index= imagesPath.find('/data/')
    replacedString = imagesPath[index+5:]
    testPath = imagesPath.replace(replacedString, '/crossValidation/' + flag + '/test/')
    df = pd.read_csv(imagesPath.replace('/images/', '/annotation.csv'))
    if not os.path.exists(testPath + 'images/'):
        os.makedirs(testPath + 'images/')
    for id in testIdList:
        if flag == 'hip':
            hipPath = imagesPath + id + 'hip.jpg'
            shutil.copy(hipPath, testPath + 'images/')
            hipImageName = id + 'hip.jpg'
            testImageNameList.append(hipImageName)
        if flag == 'knee':
            kneePath = imagesPath + id + 'knee.jpg'
            shutil.copy(kneePath, testPath + 'images/')
            kneeImageName = id + 'knee.jpg'
            testImageNameList.append(kneeImageName)
        if flag == 'ankle':
            anklePath = imagesPath + id + 'ankle.jpg'
            shutil.copy(anklePath, testPath + 'images/')
            ankleImageName = id + 'ankle.jpg'
            testImageNameList.append(ankleImageName)

    testDf = df[df.imgName.isin(testImageNameList)]
    testDf.to_csv(testPath + 'annotation.csv', index=False)


imagesPath = '/data/images/'

idList = extractPatientID(imagesPath)
tmpTrainIdList, testIdList = devideTrainAndTest(idList)


hipImagesPath = 'hipKneeAnkleImagesAndAnnotation/hip/images/'
kneeImagesPath = 'hipKneeAnkleImagesAndAnnotation/knee/images/'
ankleImagesPath = 'hipKneeAnkleImagesAndAnnotation/ankle/images/'
fiveCrossValidation(tmpTrainIdList, hipImagesPath, flag='hip')
fiveCrossValidation(tmpTrainIdList, kneeImagesPath, flag='knee')
fiveCrossValidation(tmpTrainIdList, ankleImagesPath, flag='ankle')

testData(testIdList, hipImagesPath, flag='hip')
testData(testIdList, kneeImagesPath, flag='knee')
testData(testIdList, ankleImagesPath, flag='ankle')
