import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import math
import cv2
import scipy.stats

def calculateAngle(hipCoor, kneeCoor, ankleCoor, leftOrRight):
    ang = math.degrees(
        math.atan2(ankleCoor[1] - kneeCoor[1], ankleCoor[0] - kneeCoor[0]) - math.atan2(hipCoor[1] - kneeCoor[1],
                                                                                        hipCoor[0] - kneeCoor[0]))
    # print(ang)
    if leftOrRight == 'left':
        return ang - 180
    if leftOrRight == 'right':
        return -(ang - 180)
def findCircle(x1, y1, x2, y2, x3, y3):
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2);

    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2);

    sx21 = pow(x2, 2) - pow(x1, 2);
    sy21 = pow(y2, 2) - pow(y1, 2);

    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
                              ((y31) * (x12) - (y21) * (x13))))

    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
         (2 * ((x31) * (y12) - (x21) * (y13))))
    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1);

    centerX = -g
    centerY = -f
    sqr_of_r = centerX * centerX + centerY * centerY - c;

    # r is the radius
    r = int(round(math.sqrt(sqr_of_r), 5))
    return int(round(centerX, 0)), int(round(centerY, 0)), r

def findCenter(x1, y1, x2, y2):
    centerX = int((x1 + x2) / 2)
    centerY = int((y1 + y2) / 2)
    distance = math.sqrt((centerX - x1) ** 2 + (centerY - y1) ** 2)
    return centerX, centerY, distance

def extractPointCoor(cleanedDf, idx):
    pointInfo = cleanedDf.iloc[idx, 5]
    pointXIdx = pointInfo.find('cx')
    pointYidx = pointInfo.find('cy')
    pointXCoor = int(pointInfo[pointXIdx + 4: pointYidx - 2])
    pointYCoor = int(pointInfo[pointYidx + 4: -1])
    return pointXCoor, pointYCoor

def generateTxtAnnotation(cleanCsvPath):
    cleanedDf = pd.read_csv(cleanCsvPath)
    leftAngleList = []
    rightAngleList = []
    rowNum = cleanedDf.shape[0]
    print('start')
    for rowIdx in range(0, rowNum, 14):
        print('hhh')
        imgId = cleanedDf.iloc[rowIdx, 0]
    

        # left hip
        x1, y1 = extractPointCoor(cleanedDf, rowIdx)
        x2, y2 = extractPointCoor(cleanedDf, rowIdx + 1)
        x3, y3 = extractPointCoor(cleanedDf, rowIdx + 2)
        leftHipCenter = findCircle(x1, y1, x2, y2, x3, y3)

        # right hip
        x4, y4 = extractPointCoor(cleanedDf, rowIdx + 3)
        x5, y5 = extractPointCoor(cleanedDf, rowIdx + 4)
        x6, y6 = extractPointCoor(cleanedDf, rowIdx + 5)
        rightHipCenter = findCircle(x4, y4, x5, y5, x6, y6)

        # left knee
        kx7, ky7 = extractPointCoor(cleanedDf, rowIdx + 6)
        kx8, ky8 = extractPointCoor(cleanedDf, rowIdx + 7)
        leftKneeCenter = findCenter(kx7, ky7, kx8, ky8)

        # right knee
        kx9, ky9 = extractPointCoor(cleanedDf, rowIdx + 8)
        kx10, ky10 = extractPointCoor(cleanedDf, rowIdx + 9)
        rightKneeCenter = findCenter(kx9, ky9, kx10, ky10)

        # left ankle
        ax11, ay11 = extractPointCoor(cleanedDf, rowIdx + 10)
        ax12, ay12 = extractPointCoor(cleanedDf, rowIdx + 11)
        leftAnkleCenter = findCenter(ax11, ay11, ax12, ay12)

        # left ankle
        ax13, ay13 = extractPointCoor(cleanedDf, rowIdx + 12)
        ax14, ay14 = extractPointCoor(cleanedDf, rowIdx + 13)
        rightAnkleCenter = findCenter(ax13, ay13, ax14, ay14)

        leftAngleList.append(calculateAngle(leftHipCenter, leftKneeCenter, leftAnkleCenter, 'left'))
        rightAngleList.append(calculateAngle(rightHipCenter, rightKneeCenter, rightAnkleCenter, 'right'))


    return leftAngleList, rightAngleList

def plotHist(leftAngleList, rightAngleList):
    meanLeft, varLeft = scipy.stats.distributions.norm.fit(leftAngleList)
    meanRight, varRight = scipy.stats.distributions.norm.fit(rightAngleList)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].hist(leftAngleList, bins='auto', label='varus < 0, valgus > 0 \n ' + 'mean' + str(round(meanLeft, 3)) + '    std' + str(
                   round(varLeft, 3)))
    ax[0].set_title('OAI dataset left leg statistics')
    ax[0].set_xlabel('angle value')
    ax[0].set_ylabel('number of patients')
    ax[0].legend()

    ax[1].hist(rightAngleList, bins='auto', label='varus < 0, valgus > 0 \n ' + 'mean' + str(round(meanRight, 3)) + '    std' + str(
                   round(varRight, 3)))
    ax[1].set_title('OAI dataset right leg statistics')
    ax[1].set_xlabel('angle value')
    ax[1].set_ylabel('number of patients')
    ax[1].legend()

    fig.savefig('./OAIDatasetAngleDistribution.jpg')
    plt.subplots_adjust(wspace=1)
    plt.close()

cleanCsvPath = ''
print('lol')
leftAngleList, rightAngleList = generateTxtAnnotation(cleanCsvPath)
plotHist(leftAngleList, rightAngleList)
