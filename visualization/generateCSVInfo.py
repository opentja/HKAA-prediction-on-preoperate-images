import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd
import glob
import shutil
import os
import matplotlib.pyplot as plt
import scipy.stats


def filter(idList, dateList, df):
    # read npy file then check it with tka csv data,
    # if surgery date of patients in tka csv is smaller than study date,
    # then delete that patient in csv
    deletedIdxList = []
    deltedPatientIDList = []

    for index, row in df.iterrows():
        month = row['SURGERY_DATE'].split('/')[0]
        day = row['SURGERY_DATE'].split('/')[1]
        year = row['SURGERY_DATE'].split('/')[2]
        patientID = row['CLINIC']
        tempIdx = idList.index(str(patientID))
        tmpMonth = dateList[tempIdx][4:6]
        tmpDay = dateList[tempIdx][6:8]
        tmpYear = dateList[tempIdx][0:4]
        if year > tmpYear:
            continue
        elif year < tmpYear:
            deletedIdxList.append(index)
            deltedPatientIDList.append(patientID)
        elif year == tmpYear:
            if month > tmpMonth:
                continue
            elif month < tmpMonth:
                deletedIdxList.append(index)
                deltedPatientIDList.append(patientID)
            elif month == tmpMonth:
                if day >= tmpDay:
                    continue
                else:
                    deletedIdxList.append(index)
                    deltedPatientIDList.append(patientID)
    return deletedIdxList, deltedPatientIDList


def combineInfo(npyPath, csvPath):
    maleStudyDateList = np.load(npyPath + 'maleStudyDateList.npy').tolist()
    femaleStudyDateList = np.load(npyPath + 'femaleStudyDateList.npy').tolist()
    newMalePatientIDList = np.load(npyPath + 'newMalePatientIDList.npy').tolist()
    newFemalePatientIDList = np.load(npyPath + 'newFemalePatientIDList.npy').tolist()
    # print('patients number: ', len(newFemalePatientIDList + newMalePatientIDList))
    registeredTKA = pd.read_csv(csvPath)
    newCSV = registeredTKA[['CLINIC', 'GENDER', 'SURGERY_DATE', 'BIRTH_DATE', 'DEATH_DATE', 'AGE_AT_SURGERY', 'HEIGHT',
                            'WEIGHT', 'IMAGE_DATE', 'EXAMDATE', 'DIAGNOSIS_DESCRIPTION1', 'DIAGNOSIS_DESCRIPTION2',
                            'DIAGNOSIS_DESCRIPTION3', 'DIAGNOSIS_DESCRIPTION4', 'DIAGNOSIS_DESCRIPTION5',
                            'DIAGNOSIS_DESCRIPTION6', 'DIAGNOSIS_DESCRIPTION7', 'DIAGNOSIS_DESCRIPTION8']]
    newCSV = newCSV[newCSV.CLINIC.isin(newMalePatientIDList + newFemalePatientIDList)]
    newCSV.index = range(len(newCSV.index))

    deletedIdxList, deletedPatientIDList = filter(newMalePatientIDList + newFemalePatientIDList,
                                                  maleStudyDateList + femaleStudyDateList, newCSV)
    rows = newCSV.index[deletedIdxList]
    newCSV.drop(rows, inplace=True)
    newCSV.to_csv('new.csv', index=False)

    print(deletedPatientIDList)
    return deletedPatientIDList


# let dict to store information
# {patientid: [index], patientid:[index]}

def moveImages(preoperate, postoperate, deletedIDList):
    for imageName in glob.glob(preoperate + '*.jpg'):
        studyDateIdx = imageName.find('studyDate')
        idIndex = imageName.find('patientID')
        patientID = int(imageName[idIndex + 9:studyDateIdx])
        if patientID in deletedIDList:
            print(patientID)
            shutil.move(imageName, postoperate)


def secFilter():
    # if patient has two or more surgery record, only keep the smallest one
    df = pd.read_csv('./new.csv')
    duplicatedIDList = []
    idDF = df['CLINIC']
    tmp = df[idDF.duplicated()]
    duplicatedIndex = []
    for index, row in tmp.iterrows():
        duplicatedIDList.append(row['CLINIC'])
    for patientID in duplicatedIDList:
        indexList = df.index[df['CLINIC'] == patientID].tolist()
        dateList = df[df['CLINIC'] == patientID]['SURGERY_DATE'].tolist()
        for idx in range(len(dateList)):
            tmpDate = dateList[idx].replace('/', '')
            tmpDay = tmpDate[0:4]
            tmpYear = tmpDate[4:]
            dateList[idx] = int(tmpYear + tmpDay)
        maxDateIndex = dateList.index(max(dateList))
        duplicatedIndex.append(indexList[maxDateIndex])
    # delete patient infor by index list
    update_df = df.drop(duplicatedIndex)
    print(update_df)
    update_df.to_csv('second.csv', index=False)


def thirdFilter(npyPath):
    maleLeftAnglePredList = np.load(npyPath + 'maleLeftAnglePredList.npy').tolist()
    maleRightAnglePredList = np.load(npyPath + 'maleRightAnglePredList.npy').tolist()
    femaleLeftAnglePredList = np.load(npyPath + 'femaleLeftAnglePredList.npy').tolist()
    femaleRightAnglePredList = np.load(npyPath + 'femaleRightAnglePredList.npy').tolist()
    maleStudyDateList = np.load(npyPath + 'maleStudyDateList.npy').tolist()
    femaleStudyDateList = np.load(npyPath + 'femaleStudyDateList.npy').tolist()
    newMalePatientIDList = np.load(npyPath + 'newMalePatientIDList.npy').tolist()
    newFemalePatientIDList = np.load(npyPath + 'newFemalePatientIDList.npy').tolist()

    mfList = newMalePatientIDList + newFemalePatientIDList
    mfDateList = maleStudyDateList + femaleStudyDateList
    mfLeftAngleList = maleLeftAnglePredList + femaleLeftAnglePredList
    mfRightAngleList = maleRightAnglePredList + femaleRightAnglePredList
    newMFDateList = []
    newMFLeftAngleList = []
    newMFRightAngleList = []
    idNameDict = {}
    jpgNameList = os.listdir(npyPath + 'legImages/')
    df = pd.read_csv('./second.csv')
    for jpgName in jpgNameList:
        dateIdx = jpgName.find('studyDate')
        patientID = jpgName[9:dateIdx]
        idNameDict[patientID] = jpgName
    newJpgNameList = []
    deletedID = []
    for _, row in df.iterrows():
        csvID = str(row['CLINIC'])
        if csvID in idNameDict.keys():
            listIdx = mfList.index(csvID)
            newMFDateList.append(mfDateList[listIdx])
            newMFLeftAngleList.append(mfLeftAngleList[listIdx])
            newMFRightAngleList.append(mfRightAngleList[listIdx])
            newJpgNameList.append(idNameDict[csvID])
        else:
            deletedID.append(csvID)
    df = df[~df['CLINIC'].isin(deletedID)]
    df['studyDate'] = newMFDateList
    df['leftLegAngle'] = newMFLeftAngleList
    df['rightLegAngle'] = newMFRightAngleList
    df['imageName'] = newJpgNameList
    df.to_csv('final.csv', index=False)


def filerOutlier(outlierPath):
    biggerThan25 = outlierPath + 'biggerThan25/'
    biggerThan20 = outlierPath + 'biggerThan20/'
    biggerThan15 = outlierPath + 'biggerThan15/'
    biggerThan10 = outlierPath + 'biggerThan10/'
    biggerThan5 = outlierPath + 'biggerThan5/'
    if not os.path.exists(biggerThan25):
        os.makedirs(biggerThan25)
    if not os.path.exists(biggerThan20):
        os.makedirs(biggerThan20)
    if not os.path.exists(biggerThan15):
        os.makedirs(biggerThan15)
    if not os.path.exists(biggerThan10):
        os.makedirs(biggerThan10)
    if not os.path.exists(biggerThan5):
        os.makedirs(biggerThan5)

    # extract bigger than 25
    df = pd.read_csv('./final.csv')
    newCSV = df[(df['leftLegAngle'] < -25) | (df['leftLegAngle'] > 25) | (df['rightLegAngle'] < -25) | (
                df['rightLegAngle'] > 25)]
    imageNameList = newCSV['imageName'].tolist()
    for imageName in imageNameList:
        shutil.move(npyPath + 'legImages/' + imageName, biggerThan25)
    df0025 = df[~df['imageName'].isin(imageNameList)]
    df0025.to_csv('./smallerThan25.csv', index=False)

    # extract bigger than20
    newCSV = df0025[(df0025['leftLegAngle'] < -20) | (df0025['leftLegAngle'] > 20) | (df0025['rightLegAngle'] < -20) | (
            df0025['rightLegAngle'] > 20)]
    imageNameList = newCSV['imageName'].tolist()
    for imageName in imageNameList:
        shutil.move(npyPath + 'legImages/' + imageName, biggerThan20)
    df0020 = df0025[~df0025['imageName'].isin(imageNameList)]
    df0020.to_csv('./smallerThan20.csv', index=False)

    # extract bigger 15
    newCSV = df0020[(df0020['leftLegAngle'] < -15) | (df0020['leftLegAngle'] > 15) | (df0020['rightLegAngle'] < -15) | (
            df0020['rightLegAngle'] > 15)]
    imageNameList = newCSV['imageName'].tolist()
    for imageName in imageNameList:
        shutil.move(npyPath + 'legImages/' + imageName, biggerThan15)
    df0015 = df0020[~df0020['imageName'].isin(imageNameList)]
    df0015.to_csv('./smallerThan15.csv', index=False)

    # exitract bigger than 10
    newCSV = df0015[(df0015['leftLegAngle'] < -10) | (df0015['leftLegAngle'] > 10) | (df0015['rightLegAngle'] < -10) | (
            df0015['rightLegAngle'] > 10)]
    imageNameList = newCSV['imageName'].tolist()
    for imageName in imageNameList:
        shutil.move(npyPath + 'legImages/' + imageName, biggerThan10)
    df0010 = df0015[~df0015['imageName'].isin(imageNameList)]
    df0010.to_csv('./smallerThan10.csv', index=False)

    # extract bigger than 5
    newCSV = df0010[(df0010['leftLegAngle'] < -5) | (df0010['leftLegAngle'] > 5) | (df0010['rightLegAngle'] < -5) | (
            df0010['rightLegAngle'] > 5)]
    imageNameList = newCSV['imageName'].tolist()
    df0005 = df0010[~df0010['imageName'].isin(imageNameList)]
    df0005.to_csv('./smallerThan5.csv', index=False)

    # duplicatedInfoDict[patientID] = indexList


def plot(maleLeftAnglePredList, maleRightAnglePredList, femaleLeftAnglePredList, femaleRightAnglePredList, outlierPath,
         threshold):
    if threshold == False:
        threshold = 'infinity'
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].hist(maleLeftAnglePredList, bins='auto', label='varus < 0, valgus > 0')
    ax[0].set_title('male left leg statistics' + 'absolute value smaller than' + str(threshold) + 'degree')
    ax[0].set_xlabel('angle value')
    ax[0].set_ylabel('number of patients')
    ax[0].legend()

    ax[1].hist(maleRightAnglePredList, bins='auto', label='varus < 0, valgus > 0')
    ax[1].set_title('male right leg statistics' + 'absolute value smaller than' + str(threshold) + 'degree')
    ax[1].set_xlabel('angle value')
    ax[1].set_ylabel('number of patients')
    ax[1].legend()

    fig.savefig(outlierPath + '/' + str(threshold) + 'maleAngleDistribution.jpg')
    plt.subplots_adjust(wspace=1)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].hist(femaleLeftAnglePredList, bins='auto', label='varus < 0, valgus > 0')
    ax[0].set_title('female left leg statistics' + 'absolute value smaller than' + str(threshold) + 'degree')
    ax[0].set_xlabel('angle value')
    ax[0].set_ylabel('number of patients')
    ax[0].legend()

    ax[1].hist(femaleRightAnglePredList, bins='auto', label='varus < 0, valgus > 0')
    ax[1].set_title('female right leg statistics' + 'absolute value smaller than' + str(threshold) + 'degree')
    ax[1].set_xlabel('angle value')
    ax[1].set_ylabel('number of patients')
    ax[1].legend()
    fig.savefig(outlierPath + '/' + str(threshold) + 'femaleAngleDistribution.jpg')
    plt.subplots_adjust(wspace=1)
    plt.close()


def newFrequencyFigure(outlierPath, threshold):
    if threshold == False:
        df = pd.read_csv('./final.csv')
    else:
        df = pd.read_csv('./smallerThan' + str(threshold) + '.csv')

    maledf = df[df['GENDER'] == 'MALE']

    maleLeftAnglePredList = maledf['leftLegAngle'].tolist()
    maleRightAnglePredList = maledf['rightLegAngle'].tolist()
    femaledf = df[df['GENDER'] == 'FEMALE']
    femaleLeftAnglePredList = femaledf['leftLegAngle'].tolist()
    femaleRightAnglePredList = femaledf['rightLegAngle'].tolist()
    plot(maleLeftAnglePredList, maleRightAnglePredList, femaleLeftAnglePredList, femaleRightAnglePredList, outlierPath,
         threshold)


def getGaussianDistribution():
    df15 = pd.read_csv('./smallerThan15.csv')
    maleLeft = df15[df15['GENDER'] == 'MALE']['leftLegAngle'].tolist()
    maleRight = df15[df15['GENDER'] == 'MALE']['rightLegAngle'].tolist()
    femaleLeft = df15[df15['GENDER'] == 'FEMALE']['leftLegAngle'].tolist()
    femaleRight = df15[df15['GENDER'] == 'FEMALE']['rightLegAngle'].tolist()

    maleMeanLeft, maleVarLeft = scipy.stats.distributions.norm.fit(maleLeft)
    maleMeanRight, maleVarRight = scipy.stats.distributions.norm.fit(maleRight)
    femaleMeanLeft, femaleVarLeft = scipy.stats.distributions.norm.fit(femaleLeft)
    femaleMeanRight, femaleVarRight = scipy.stats.distributions.norm.fit(femaleRight)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].hist(maleLeft, bins='auto',
               label='varus < 0, valgus > 0 \n ' + 'mean' + str(round(maleMeanLeft, 3)) + '   std' + str(
                   round(maleVarLeft, 3)))
    ax[0].set_title('male left leg statistics' + 'absolute value smaller than' + str(15) + 'degree')
    ax[0].set_xlabel('angle value')
    ax[0].set_ylabel('number of patients')
    ax[0].legend()

    fitted_data = scipy.stats.distributions.norm.pdf(maleRight, maleMeanRight, maleVarRight)
    ax[1].hist(maleRight, bins='auto',
               label='varus < 0, valgus > 0 \n ' + 'mean' + str(round(maleMeanRight, 3)) + '    std' + str(
                   round(maleVarRight, 3)))
    ax[1].set_title('male right leg statistics' + 'absolute value smaller than' + str(15) + 'degree')
    ax[1].set_xlabel('angle value')
    ax[1].set_ylabel('number of patients')
    ax[1].legend()

    fig.savefig(outlierPath + '/' + str(15) + 'maleAngleDistribution.jpg')
    plt.subplots_adjust(wspace=1)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].hist(femaleLeft, bins='auto',
               label='varus < 0, valgus > 0 \n ' + 'mean' + str(round(femaleMeanLeft, 3)) + '    std' + str(
                   round(femaleVarLeft, 3)))
    ax[0].set_title('female left leg statistics' + 'absolute value smaller than' + str(15) + 'degree')
    ax[0].set_xlabel('angle value')
    ax[0].set_ylabel('number of patients')
    ax[0].legend()

    ax[1].hist(femaleRight, bins='auto',
               label='varus < 0, valgus > 0 \n ' + 'mean' + str(round(femaleMeanRight, 3)) + '    std' + str(
                   round(femaleVarRight, 3)))
    ax[1].set_title('female right leg statistics' + 'absolute value smaller than' + str(15) + 'degree')
    ax[1].set_xlabel('angle value')
    ax[1].set_ylabel('number of patients')
    ax[1].legend()
    fig.savefig(outlierPath + '/' + str(15) + 'femaleAngleDistribution.jpg')
    plt.subplots_adjust(wspace=1)
    plt.close()


npyPath = '/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/'
xlsxPath = '/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/PTKA.1985.2017.csv'
preoperate = '/infodev1/phi-data/shi/kneeX-ray/preoperate/ankleToHip/'
postoperate = '/infodev1/phi-data/shi/kneeX-ray/preoperate/'

deletedPatientIDList = combineInfo(npyPath, xlsxPath)
# moveImages(preoperate, postoperate, deletedPatientIDList)
secFilter()
secFilter()
thirdFilter(npyPath)

outlierPath = '/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/'
#
filerOutlier(outlierPath)
newFrequencyFigure(outlierPath, False)
newFrequencyFigure(outlierPath, 25)
newFrequencyFigure(outlierPath, 20)
newFrequencyFigure(outlierPath, 15)
newFrequencyFigure(outlierPath, 10)
getGaussianDistribution()
