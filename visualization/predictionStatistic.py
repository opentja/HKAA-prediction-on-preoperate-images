import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
import cv2
import numpy as np
import pydicom
import pandas as pd

import time



def oneDicom(dicomFilePath):
    ds = pydicom.read_file(dicomFilePath)
    sex = ds.PatientSex
    birthDate = ds.PatientBirthDate
    studyDate = ds.StudyDate

    return sex, studyDate, birthDate

def calculateAge(birthDate, currentDate):
    age = int(currentDate[0:4]) - int(birthDate[0:4]) - ((int(currentDate[4:6]), int(currentDate[6:])) < (int(birthDate[4:6]), int(birthDate[6:])))
    return int(age)

def extractPatientInfoSaveCSV(allDicomFolder, predictedFolderPath):
    patientIDList, sexList, studyDateList, birthDateList, ageWhenStudyList = [], [], [], [], []
    malePatientIDList= []
    femalePatientIDList = []
    maleStudyDateList = []
    femaleStudyDateList = []

    imageNameList = os.listdir(predictedFolderPath)
    for imageName in imageNameList:
        studyDateIdx = imageName.find('studyDate')
        nameIdx = imageName.find('name')
        patientID = imageName[9:studyDateIdx]
        name = imageName[nameIdx+4:].replace('.jpg', '.dcm')
        dicomPath = allDicomFolder+patientID+'/'+name
        sex, studyDate, birthDate = oneDicom(dicomPath)
        age = calculateAge(birthDate, studyDate)
        patientIDList.append(patientID)
        sexList.append(sex)
        studyDateList.append(studyDate)
        birthDateList.append(birthDate)
        ageWhenStudyList.append(age)
        if sex == 'M':
            malePatientIDList.append(patientID)
            maleStudyDateList.append(studyDate)
        if sex == 'F':
            femalePatientIDList.append(patientID)
            femaleStudyDateList.append(studyDate)

        # print(sex)
    patientData = {'patientID': patientIDList,'sex':sexList, 'age':ageWhenStudyList, 'studyDate':studyDateList, 'birthDate':birthDate }
    patientDf = pd.DataFrame(patientData)
    patientDf.to_csv(predictedFolderPath.replace('preoperate/ankleToHip/','preoperatePrediction/')+'patientInfo.csv', index=False)
    np.save('/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/malePatientIDList.npy', malePatientIDList)
    np.save('/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/femalePatientIDList.npy', femalePatientIDList)

def readCoorAsMap(txtPath):
    coorDict = {}
    txtFile = open(txtPath)
    coorList = txtFile.readlines()
    for index in range(len(coorList)):
        parts = coorList[index].replace('\n', '').split(',')
        imageName = parts[0]
        coordinates = parts[1:]
        coordinates = (list(map(int, coordinates)))
        coorDict[imageName] = coordinates
    return coorDict

def calculateAngle(hipCoor, kneeCoor, ankleCoor, leftOrRight):
    ang = math.degrees(
        math.atan2(ankleCoor[1] - kneeCoor[1], ankleCoor[0] - kneeCoor[0]) - math.atan2(hipCoor[1] - kneeCoor[1],
                                                                                        hipCoor[0] - kneeCoor[0]))
    # print(ang)
    if leftOrRight == 'left':
        return ang - 180
    if leftOrRight == 'right':
        return -(ang - 180)

def markPointsAndLine(imageArr, hipCoors, kneeCoors, ankleCoors, wholeImagePath):
    leftHipPred = (hipCoors[0], hipCoors[1])
    rightHipPred = (hipCoors[2], hipCoors[3])

    leftKneePred = (kneeCoors[0], kneeCoors[1])
    rightKneePred = (kneeCoors[2], kneeCoors[3])

    leftAnklePred = (ankleCoors[0], ankleCoors[1])
    rightAnklePred = (ankleCoors[2], ankleCoors[3])

    pointRadius = 12
    redColor = (0, 0, 255)
    thickness = -1
    # draw points
    imageArr = cv2.circle(imageArr, leftHipPred, pointRadius, redColor, thickness)
    imageArr = cv2.circle(imageArr, leftKneePred, pointRadius, redColor, thickness)
    imageArr = cv2.circle(imageArr, leftAnklePred, pointRadius, redColor, thickness)

    imageArr = cv2.circle(imageArr, rightHipPred, pointRadius, redColor, thickness)
    imageArr = cv2.circle(imageArr, rightKneePred, pointRadius, redColor, thickness)
    imageArr = cv2.circle(imageArr, rightAnklePred, pointRadius, redColor, thickness)

    # draw red line among prediction points
    cv2.line(imageArr, leftHipPred, leftKneePred, (0, 0, 255), 2)
    cv2.line(imageArr, leftKneePred, leftAnklePred, (0, 0, 255), 2)

    cv2.line(imageArr, rightHipPred, rightKneePred, (0, 0, 255), 2)
    cv2.line(imageArr, rightKneePred, rightAnklePred, (0, 0, 255), 2)

    # angle prediction
    leftAnglePred = round(calculateAngle(leftHipPred, leftKneePred, leftAnklePred, 'left'), 3)
    rightAnglePred = round(calculateAngle(rightHipPred, rightKneePred, rightAnklePred, 'right'), 3)



    tmpIndex = wholeImagePath.rfind('/')
    imageName = wholeImagePath[tmpIndex + 1:]

    #write angle into txt file
    resultTxt = open(wholeImagePath[:tmpIndex + 1].replace('legImages/', '') + 'angleResult.txt', 'a+')
    resultTxt.write(
        imageName +  ',' + str(leftAnglePred) + ',' + str(
            rightAnglePred) + '\n')
    resultTxt.close()

    # add text on image

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w, _ = imageArr.shape
    org1 = (int(w / 2), int(h / 4))
    org2 = (int(w / 2), 3 * int(h / 4))

    # fontScale
    fontScale = 2
    # green color in BGR
    greenColor = (0, 255, 0)
    # Line thickness of 2 px
    thickness = 2
    degreeSign = ''
    # Using cv2.putText() method
    imageArr = cv2.putText(imageArr,
                           'right predicted angle:' + str(round(leftAnglePred, 2)) + degreeSign,
                           (50, 100),
                           font, fontScale, greenColor, thickness)

    imageArr = cv2.putText(imageArr,
                           'left predicted angle:' + str(round(rightAnglePred, 2)) + degreeSign,
                           (50, 200),
                           font, fontScale, greenColor, thickness)

    cv2.imwrite(wholeImagePath, imageArr)

    return leftAnglePred, rightAnglePred

def plotImageAndAngle(hipDict, kneeDict, ankleDict, testImagePath, npyFolderPath):
    malePatientIDList = np.load(npyFolderPath+ 'malePatientIDList.npy')
    femalePatientIDList = np.load(npyFolderPath+'femalePatientIDList.npy')
    newMalePatientIDList = []
    newFemalePatientIDList = []
    maleStudyDateList = []
    femaleStudyDateList = []
    if not os.path.exists(testImagePath  + 'legImages/'):
        os.makedirs(testImagePath  + 'legImages/')
    maleLeftAnglePredList, maleRightAnglePredList = [], []
    femaleLeftAnglePredList, femaleRightAnglePredList = [], []
    for imageName, coordinates in hipDict.items():
        hipCoors = coordinates
        kneeCoors = kneeDict[imageName.replace('hip', 'knee')]
        ankleCoors = ankleDict[imageName.replace('hip', 'ankle')]

        hipImagePath = testImagePath + 'hip/imagesWithEpochs48Lr0.001BatchSize16Resolution256/' + imageName
        kneeImagePath = testImagePath + 'knee/imagesWithEpochs48Lr0.001BatchSize8Resolution256/' + imageName.replace(
            'hip', 'knee')
        ankleImagePath = testImagePath + 'ankle/imagesWithEpochs48Lr0.001BatchSize8Resolution256/' + imageName.replace(
            'hip', 'ankle')

        hipImageArr = cv2.imread(hipImagePath)
        kneeImageArr = cv2.imread(kneeImagePath)
        ankleImageArr = cv2.imread(ankleImagePath)

        hipY, _ = hipImageArr.shape[:2]
        kneeY, _ = kneeImageArr.shape[:2]

        kneeCoors[1] = kneeCoors[1] + hipY
        kneeCoors[3] = kneeCoors[3] + hipY

        ankleCoors[1] = ankleCoors[1] + hipY + kneeY
        ankleCoors[3] = ankleCoors[3] + hipY + kneeY

        wholeImageArr = np.concatenate((hipImageArr, kneeImageArr, ankleImageArr), axis=0)
        wholeImagePath = testImagePath + 'legImages/' + imageName.replace('hip', '')
        studyDateIdx = imageName.find('studyDate')
        nameIdx = imageName.find('name')
        studyDate = imageName[studyDateIdx+9:nameIdx]
        # print(studyDate)
        patientID = imageName[9:studyDateIdx]

        if patientID in malePatientIDList:
            maleLeftAnglePred, maleRightAnglePred = markPointsAndLine(wholeImageArr, hipCoors,kneeCoors, ankleCoors, wholeImagePath)
            maleLeftAnglePredList.append(maleLeftAnglePred)
            maleRightAnglePredList.append(maleRightAnglePred)
            newMalePatientIDList.append(patientID)
            maleStudyDateList.append(studyDate)
        if patientID in femalePatientIDList:
            femaleLeftAnglePred, femaleRightAnglePred = markPointsAndLine(wholeImageArr, hipCoors, kneeCoors, ankleCoors,
                                                                      wholeImagePath)
            femaleLeftAnglePredList.append(femaleLeftAnglePred)
            femaleRightAnglePredList.append(femaleRightAnglePred)
            newFemalePatientIDList.append(patientID)
            femaleStudyDateList.append(studyDate)

    np.save(testImagePath + 'maleLeftAnglePredList.npy', maleLeftAnglePredList)
    np.save(testImagePath + 'maleRightAnglePredList.npy', maleRightAnglePredList)
    np.save(testImagePath + 'femaleLeftAnglePredList.npy', femaleLeftAnglePredList)
    np.save(testImagePath + 'femaleRightAnglePredList.npy', femaleRightAnglePredList)
    np.save(testImagePath + 'newMalePatientIDList.npy', newMalePatientIDList)
    np.save(testImagePath + 'newFemalePatientIDList.npy', newFemalePatientIDList)
    np.save(testImagePath + 'maleStudyDateList.npy', maleStudyDateList)
    np.save(testImagePath + 'femaleStudyDateList.npy', femaleStudyDateList)


def plotStatisticalFigure(npyFilePath):
    maleLeftAnglePredList = np.load(npyFilePath + 'maleLeftAnglePredList.npy')
    maleRightAnglePredList = np.load(npyFilePath + 'maleRightAnglePredList.npy')
    femaleLeftAnglePredList = np.load(npyFilePath + 'femaleLeftAnglePredList.npy')
    femaleRightAnglePredList = np.load(npyFilePath + 'femaleRightAnglePredList.npy')

    # mxAxisMaxValue = np.arange(0, len(maleLeftAnglePredList))
    # fxAxisMaxValue = np.arange(0, len(femaleLeftAnglePredList))
    # print(xAxisMaxValue)
    # label color ----> blue
    # prediction color ---> yellow
    # valgus ---> positive value
    # varus ---> negative value
    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    ax[0].hist(maleLeftAnglePredList, bins='auto',label='varus < 0, valgus > 0')
    ax[0].set_title('male left leg statistics')
    ax[0].set_xlabel('angle value')
    ax[0].set_ylabel('number of patients')
    ax[0].legend()

    ax[1].hist(maleRightAnglePredList, bins='auto',label='varus < 0, valgus > 0')
    ax[1].set_title('male right leg statistics')
    ax[1].set_xlabel('angle value')
    ax[1].set_ylabel('number of patients')
    ax[1].legend()

    fig.savefig(npyFilePath + '/maleAngleDistribution')
    plt.subplots_adjust(wspace=1)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].hist(femaleLeftAnglePredList, bins='auto',label='varus < 0, valgus > 0')
    ax[0].set_title('female left leg statistics')
    ax[0].set_xlabel('angle value')
    ax[0].set_ylabel('number of patients')
    ax[0].legend()

    ax[1].hist(femaleRightAnglePredList, bins='auto',label='varus < 0, valgus > 0')
    ax[1].set_title('female right leg statistics')
    ax[1].set_xlabel('angle value')
    ax[1].set_ylabel('number of patients')
    ax[1].legend()
    fig.savefig(npyFilePath + '/femaleAngleDistribution')
    plt.subplots_adjust(wspace=1)
    plt.close()

start = time.time()
print("hello")

allDicomFolder='/infodev1/phi-data/shi/ankletohip/allDicomFiles/'
predictedFolderPath='/infodev1/phi-data/shi/kneeX-ray/preoperate/ankleToHip/'

extractPatientInfoSaveCSV(allDicomFolder, predictedFolderPath)

testImagePath = '/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/'

hipPath = '/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/hip/coordinateWithEpochs48Lr0.001BatchSize16Resolution256.txt'
kneePath = '/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/knee/coordinateWithEpochs48Lr0.001BatchSize8Resolution256.txt'
anklePath = '/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/ankle/coordinateWithEpochs48Lr0.001BatchSize8Resolution256.txt'
npyFolderPath = '/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/'
hipDict = readCoorAsMap(hipPath)
kneeDict = readCoorAsMap(kneePath)
ankleDict = readCoorAsMap(anklePath)

plotImageAndAngle(hipDict, kneeDict, ankleDict, testImagePath, npyFolderPath)
plotStatisticalFigure(npyFolderPath)


end = time.time()
print(end - start)


