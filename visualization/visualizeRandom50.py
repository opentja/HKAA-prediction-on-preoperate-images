import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
import cv2
import numpy as np


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


# def calculateAngle(hipCoor, kneeCoor, ankleCoor):
#     ang = math.degrees(
#         math.atan2(ankleCoor[1] - kneeCoor[1], ankleCoor[0] - kneeCoor[0]) - math.atan2(hipCoor[1] - kneeCoor[1],
#                                                                                         hipCoor[0] - kneeCoor[0]))
#     return ang + 360 if ang < 0 else ang

def calculateAngle(hipCoor, kneeCoor, ankleCoor, leftOrRight):
    ang = math.degrees(
        math.atan2(ankleCoor[1] - kneeCoor[1], ankleCoor[0] - kneeCoor[0]) - math.atan2(hipCoor[1] - kneeCoor[1],
                                                                                        hipCoor[0] - kneeCoor[0]))
    # print(ang)
    if leftOrRight == 'left':
        return ang - 180
    if leftOrRight == 'right':
        return -(ang - 180)


def pixelDiff(x1, y1, x2, y2, x3, y3, X1, Y1, X2, Y2, X3, Y3):
    distance1 = math.sqrt(((x1 - X1) ** 2) + ((y1 - Y1) ** 2))
    distance2 = math.sqrt(((x2 - X2) ** 2) + ((y2 - Y2) ** 2))
    distance3 = math.sqrt(((x3 - X3) ** 2) + ((y3 - Y3) ** 2))

    return ((distance1 + distance2 + distance3) / 2)


def markPointsAndLine(imageArr, hipCoors, kneeCoors, ankleCoors, wholeImagePath):
    leftHipLabel = (hipCoors[0], hipCoors[1])
    rightHipLabel = (hipCoors[2], hipCoors[3])

    leftHipPred = (hipCoors[4], hipCoors[5])
    rightHipPred = (hipCoors[6], hipCoors[7])

    leftKneeLabel = (kneeCoors[0], kneeCoors[1])
    rightKneeLabel = (kneeCoors[2], kneeCoors[3])

    leftKneePred = (kneeCoors[4], kneeCoors[5])
    rightKneePred = (kneeCoors[6], kneeCoors[7])

    leftAnkleLabel = (ankleCoors[0], ankleCoors[1])
    rightAnkleLabel = (ankleCoors[2], ankleCoors[3])

    leftAnklePred = (ankleCoors[4], ankleCoors[5])
    rightAnklePred = (ankleCoors[6], ankleCoors[7])

    # mark points with color, label is blue color, prediction is red color
    # mark left Hip label

    pointRadius = 12
    redColor = (0, 0, 255)
    blueColor = (255, 0, 0)
    thickness = -1
    # draw points
    imageArr = cv2.circle(imageArr, leftHipLabel, pointRadius, blueColor, thickness)
    imageArr = cv2.circle(imageArr, leftKneeLabel, pointRadius, blueColor, thickness)
    imageArr = cv2.circle(imageArr, leftAnkleLabel, pointRadius, blueColor, thickness)

    imageArr = cv2.circle(imageArr, rightHipLabel, pointRadius, blueColor, thickness)
    imageArr = cv2.circle(imageArr, rightKneeLabel, pointRadius, blueColor, thickness)
    imageArr = cv2.circle(imageArr, rightAnkleLabel, pointRadius, blueColor, thickness)

    imageArr = cv2.circle(imageArr, leftHipPred, pointRadius, redColor, thickness)
    imageArr = cv2.circle(imageArr, leftKneePred, pointRadius, redColor, thickness)
    imageArr = cv2.circle(imageArr, leftAnklePred, pointRadius, redColor, thickness)

    imageArr = cv2.circle(imageArr, rightHipPred, pointRadius, redColor, thickness)
    imageArr = cv2.circle(imageArr, rightKneePred, pointRadius, redColor, thickness)
    imageArr = cv2.circle(imageArr, rightAnklePred, pointRadius, redColor, thickness)

    # draw blue line among label points
    cv2.line(imageArr, leftHipLabel, leftKneeLabel, (255, 0, 0), 2)
    cv2.line(imageArr, leftKneeLabel, leftAnkleLabel, (255, 0, 0), 2)

    cv2.line(imageArr, rightHipLabel, rightKneeLabel, (255, 0, 0), 2)
    cv2.line(imageArr, rightKneeLabel, rightAnkleLabel, (255, 0, 0), 2)

    # draw red line among prediction points
    cv2.line(imageArr, leftHipPred, leftKneePred, (0, 0, 255), 2)
    cv2.line(imageArr, leftKneePred, leftAnklePred, (0, 0, 255), 2)

    cv2.line(imageArr, rightHipPred, rightKneePred, (0, 0, 255), 2)
    cv2.line(imageArr, rightKneePred, rightAnklePred, (0, 0, 255), 2)

    # anlge label
    leftAngleLabel = round(calculateAngle(leftHipLabel, leftKneeLabel, leftAnkleLabel, 'left'), 3)
    rightAngleLabel = round(calculateAngle(rightHipLabel, rightKneeLabel, rightAnkleLabel, 'right'), 3)

    # angle prediction
    leftAnglePred = round(calculateAngle(leftHipPred, leftKneePred, leftAnklePred, 'left'), 3)
    rightAnglePred = round(calculateAngle(rightHipPred, rightKneePred, rightAnklePred, 'right'), 3)

    tmpIndex = wholeImagePath.rfind('/')
    imageName = wholeImagePath[tmpIndex + 1:]

    #     write angle into txt file

    resultTxt = open(wholeImagePath[:tmpIndex + 1].replace('legImages/', '') + 'angleResult.txt', 'a+')
    resultTxt.write(
        imageName + ',' + str(leftAngleLabel) + ',' + str(rightAngleLabel) + ',' + str(leftAnglePred) + ',' + str(
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
    blueColor = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2

    degreeSign = ''
    # Using cv2.putText() method
    imageArr = cv2.putText(imageArr,
                           'right Angle diff:' + str(round(abs(leftAngleLabel - leftAnglePred), 2)) + degreeSign,
                           (50, 100),
                           font, fontScale, greenColor, thickness)

    imageArr = cv2.putText(imageArr,
                           'right labeled angle:' + str(round(leftAngleLabel, 2)) + degreeSign,
                           (50, 200),
                           font, fontScale, greenColor, thickness)

    imageArr = cv2.putText(imageArr,
                           'right predicted angle:' + str(round(leftAnglePred, 2)) + degreeSign,
                           (50, 300),
                           font, fontScale, greenColor, thickness)

    imageArr = cv2.putText(imageArr,
                           'left Angle diff:' + str(round(abs(rightAngleLabel - rightAnglePred), 2)) + degreeSign,
                           (50, 500),
                           font, fontScale, greenColor, thickness)

    imageArr = cv2.putText(imageArr,
                           'right labeled angle:' + str(round(rightAngleLabel, 2)) + degreeSign,
                           (50, 600),
                           font, fontScale, greenColor, thickness)

    imageArr = cv2.putText(imageArr,
                           'right predicted angle:' + str(round(rightAnglePred, 2)) + degreeSign,
                           (50, 700),
                           font, fontScale, greenColor, thickness)

    cv2.imwrite(wholeImagePath, imageArr)

    return leftAngleLabel, rightAngleLabel, leftAnglePred, rightAnglePred


def plotImageAndAngle(hipDict, kneeDict, ankleDict, testImagePath):
    if not os.path.exists(testImagePath  + '/legImages/'):
        os.makedirs(testImagePath + '/legImages/')
    leftAngleLabelList, rightAngleLabelList, leftAnglePredList, rightAnglePredList = [], [], [], []
    leftPixelDiffList, rightPixelDiffList = [], []
    leftDiffList = []
    rightDiffList = []
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
        kneeCoors[5] = kneeCoors[5] + hipY
        kneeCoors[7] = kneeCoors[7] + hipY

        ankleCoors[1] = ankleCoors[1] + hipY + kneeY
        ankleCoors[3] = ankleCoors[3] + hipY + kneeY
        ankleCoors[5] = ankleCoors[5] + hipY + kneeY
        ankleCoors[7] = ankleCoors[7] + hipY + kneeY

        wholeImageArr = np.concatenate((hipImageArr, kneeImageArr, ankleImageArr), axis=0)
        wholeImagePath = testImagePath  + '/legImages/' + imageName.replace('hip', '')
        leftAngleLabel, rightAngleLabel, leftAnglePred, rightAnglePred = markPointsAndLine(wholeImageArr, hipCoors,
                                                                                           kneeCoors, ankleCoors,
                                                                                           wholeImagePath)
        # print(imageName.replace('hip', ''))
        # print(hipCoors,kneeCoors,ankleCoors)
        # print(leftAngleLabel, leftAnglePred, rightAngleLabel, rightAnglePred, '\n')
        leftPixelDiff = pixelDiff(hipCoors[0], hipCoors[1], kneeCoors[0], kneeCoors[1], ankleCoors[0], ankleCoors[1],
                                  hipCoors[4], hipCoors[5], kneeCoors[4], kneeCoors[5], ankleCoors[4], ankleCoors[5])

        rightPixelDiff = pixelDiff(hipCoors[2], hipCoors[3], kneeCoors[2], kneeCoors[3], ankleCoors[2], ankleCoors[3],
                                   hipCoors[6], hipCoors[7], kneeCoors[6], kneeCoors[7], ankleCoors[6], ankleCoors[7])
        # print(leftPixelDiff)
        # print(rightPixelDiff, '\n')
        leftPixelDiffList.append(leftPixelDiff)
        rightPixelDiffList.append(rightPixelDiff)

        leftAngleLabelList.append(leftAngleLabel)
        rightAngleLabelList.append(rightAngleLabel)
        leftAnglePredList.append(leftAnglePred)
        rightAnglePredList.append(rightAnglePred)
        leftDiffList.append(abs(leftAnglePred - leftAngleLabel))
        rightDiffList.append(abs(rightAnglePred - rightAngleLabel))

    count1 = len([i for i in leftDiffList if i <= 1.5])
    count2 = len([i for i in rightDiffList if i <= 1.5])
    total = len(leftDiffList) + len(rightDiffList)
    smaller = count1 + count2

    print('\ntotal number of angles: ', total, '   number of angle difference smaller than 1.5 is: ', smaller, 'take',
          round(smaller / total, 3), 'percentage')

    print('\naverage angle difference of left part:', round(sum(leftDiffList) / len(leftDiffList), 3))
    print('\naverage angle difference of right part:', round(sum(rightDiffList) / len(rightDiffList), 3))
    leftPartPixelDiff = round(sum(leftPixelDiffList) / len(leftPixelDiffList), 3)
    rightPartPixelDiff = round(sum(rightPixelDiffList) / len(rightPixelDiffList), 3)
    print('\naverage pixel difference of left part: ', leftPartPixelDiff)
    print('\naverage pixel difference of right part: ', rightPartPixelDiff)
    print('\naverage pixel difference of each image: ', round((leftPartPixelDiff + rightPartPixelDiff) / 2, 3))

    # print(len(leftAngleLabelList), len(leftAnglePredList), len(rightAngleLabelList), len(rightAnglePredList))
    np.save(testImagePath  + '/leftAngleLabelList.npy', leftAngleLabelList)
    np.save(testImagePath  + '/rightAngleLabelList.npy', rightAngleLabelList)
    np.save(testImagePath  + '/leftAnglePredList.npy', leftAnglePredList)
    np.save(testImagePath  + '/rightAnglePredList.npy', rightAnglePredList)


def plotStatisticalFigure(npyFilePath):
    leftAngleLabelList = np.load(npyFilePath  + '/leftAngleLabelList.npy')
    leftAnglePredList = np.load(npyFilePath  + '/leftAnglePredList.npy')
    rightAngleLabelList = np.load(npyFilePath  + '/rightAngleLabelList.npy')
    rightAnglePredList = np.load(npyFilePath  + '/rightAnglePredList.npy')
    # print(len(leftAngleLabelList), len(leftAnglePredList), len(rightAngleLabelList), len(rightAnglePredList))

    xAxisMaxValue = np.arange(0, len(leftAngleLabelList))
    # print(xAxisMaxValue)
    # label color ----> blue
    # prediction color ---> yellow
    # valgus ---> positive value
    # varus ---> negative value
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].scatter(xAxisMaxValue, leftAngleLabelList, marker='.', color='blue', label='annotation angle')
    ax[0].scatter(xAxisMaxValue, leftAnglePredList, marker='*', color='red', label='predicted angle')
    # Hide the right and top spines
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['bottom'].set_position(('data', 0))
    ax[0].set_xlabel('image ID')
    ax[0].set_ylabel('angle value')
    ax[0].xaxis.set_label_coords(1.05, 0.5)

    ax[0].legend()

    ax[1].scatter(xAxisMaxValue, rightAngleLabelList, marker='.', color='blue', label='annotation angle')
    ax[1].scatter(xAxisMaxValue, rightAnglePredList, marker='*', color='red', label='predicted angle')
    # Hide the right and top spines
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_position(('data', 0))
    ax[1].set_xlabel('image ID')
    ax[1].set_ylabel('angle value')
    ax[1].xaxis.set_label_coords(1.05, 0.5)
    ax[1].legend()
    fig.savefig(npyFilePath  + '/angleDistribution')
    plt.subplots_adjust(wspace=1)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].scatter(leftAngleLabelList, leftAnglePredList, marker='o', color='blue', s=0.5, label='annotation angle')
    ax[0].plot(np.arange(-20, 20), np.arange(-20, 20), color='green', label='x=y')
    # Hide the right and top spines
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['bottom'].set_position(('data', 0))
    ax[0].spines['left'].set_position(('data', 0))
    ax[0].set_xlabel('labeledAngle')
    ax[0].set_ylabel('predictedAngle')
    ax[0].xaxis.set_label_coords(0.88, 0.45)
    ax[0].yaxis.set_label_coords(0.45, 0.85)
    ax[0].legend()

    ax[1].scatter(rightAngleLabelList, rightAnglePredList, marker='o', color='blue', s=0.5, label='annotation angle')
    ax[1].plot(np.arange(-20, 20), np.arange(-20, 20), color='green', label='x=y')
    # Hide the right and top spines
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_position(('data', 0))
    ax[1].spines['left'].set_position(('data', 0))
    ax[1].set_xlabel('labeledAngle')
    ax[1].set_ylabel('predictedAngle')
    ax[1].xaxis.set_label_coords(0.88, 0.45)
    ax[1].yaxis.set_label_coords(0.45, 0.85)
    ax[1].legend()
    fig.savefig(npyFilePath  + '/angleDistributionCo')
    plt.subplots_adjust(wspace=1)
    plt.close()

flagList = ['0005','0510','1015','1520','2025']
for flag in flagList:
    testImagePath = '/randomSelect50/ankleToHip'+flag+'PredRes/'
    hipPath = '/randomSelect50/ankleToHip'+flag+'PredRes/hip/coordinateWithEpochs48Lr0.001BatchSize16Resolution256.txt'
    kneePath = '/randomSelect50/ankleToHip'+flag+'PredRes/knee/coordinateWithEpochs48Lr0.001BatchSize8Resolution256.txt'
    anklePath = '/randomSelect50/ankleToHip'+flag+'PredRes/ankle/coordinateWithEpochs48Lr0.001BatchSize8Resolution256.txt'

    hipDict = readCoorAsMap(hipPath)
    kneeDict = readCoorAsMap(kneePath)
    ankleDict = readCoorAsMap(anklePath)

    plotImageAndAngle(hipDict, kneeDict, ankleDict, testImagePath)
    plotStatisticalFigure(testImagePath)
