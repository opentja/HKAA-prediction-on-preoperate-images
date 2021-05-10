import pandas as pd
import cv2
import math
import os


def extractPointCoor(cleanedDf, idx):
    pointInfo = cleanedDf.iloc[idx, 5]
    pointXIdx = pointInfo.find('cx')
    pointYidx = pointInfo.find('cy')
    pointXCoor = int(pointInfo[pointXIdx + 4: pointYidx - 2])
    pointYCoor = int(pointInfo[pointYidx + 4: -1])
    return pointXCoor, pointYCoor


def generateTxtAnnotation(cleanCsvPath, imagesPath, newImagePath):
    cleanedDf = pd.read_csv(cleanCsvPath)

    rowNum = cleanedDf.shape[0]
    x1List, x2List, x3List, x4List, x5List, x6List, x7List, x8List, x9List, x10List, x11List, x12List, x13List, x14List = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    y1List, y2List, y3List, y4List, y5List, y6List, y7List, y8List, y9List, y10List, y11List, y12List, y13List, y14List = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    hipImageNameList, kneeImageNameList, ankleImageNameList = [], [], []
    for rowIdx in range(0, rowNum, 20):
        imgId = cleanedDf.iloc[rowIdx, 0]
        imgPath = imagesPath + imgId
        # print(os.path.exists(imgPath))
        img = cv2.imread(imgPath)
        h, w, c = img.shape
        third = int(math.floor(h / 3))
        hipImg = img[0:third, :, :]
        kneeImg = img[third:2 * third, :, :]
        ankleImg = img[2 * third:, :, :]

        # left hip
        x1, y1 = extractPointCoor(cleanedDf, rowIdx)
        x2, y2 = extractPointCoor(cleanedDf, rowIdx + 1)
        x3, y3 = extractPointCoor(cleanedDf, rowIdx + 2)

        # right hip
        x4, y4 = extractPointCoor(cleanedDf, rowIdx + 3)
        x5, y5 = extractPointCoor(cleanedDf, rowIdx + 4)
        x6, y6 = extractPointCoor(cleanedDf, rowIdx + 5)

        # left knee
        kx7, ky7 = extractPointCoor(cleanedDf, rowIdx + 6)
        kx8, ky8 = extractPointCoor(cleanedDf, rowIdx + 7)
        x7, y7 = kx7, ky7 - third
        x8, y8 = kx8, ky8 - third

        # right knee
        kx9, ky9 = extractPointCoor(cleanedDf, rowIdx + 8)
        kx10, ky10 = extractPointCoor(cleanedDf, rowIdx + 9)
        x9, y9 = kx9, ky9 - third
        x10, y10 = kx10, ky10 - third

        # left ankle
        ax11, ay11 = extractPointCoor(cleanedDf, rowIdx + 10)
        ax12, ay12 = extractPointCoor(cleanedDf, rowIdx + 11)
        x11, y11 = ax11, ay11 - 2 * third
        x12, y12 = ax12, ay12 - 2 * third

        # left ankle
        ax13, ay13 = extractPointCoor(cleanedDf, rowIdx + 12)
        ax14, ay14 = extractPointCoor(cleanedDf, rowIdx + 13)
        x13, y13 = ax13, ay13 - 2 * third
        x14, y14 = ax14, ay14 - 2 * third
        if y1 < third and y2 < third and y3 < third and y3 < third and y3 < third and y3 < third:
            # save new image and new annotations
            hipImageName = imgId[:-4] + 'hip' + '.jpg'
            if not os.path.exists(newImagePath + 'hip/images/'):
                os.makedirs(newImagePath + 'hip/images/')
            cv2.imwrite(newImagePath + 'hip/images/' + hipImageName, hipImg)

            kneeImageName = imgId[:-4] + 'knee' + '.jpg'
            if not os.path.exists(newImagePath + 'knee/images/'):
                os.makedirs(newImagePath + 'knee/images/')
            cv2.imwrite(newImagePath + 'knee/images/' + kneeImageName, kneeImg)

            ankleImageName = imgId[:-4] + 'ankle' + '.jpg'
            if not os.path.exists(newImagePath + 'ankle/images/'):
                os.makedirs(newImagePath + 'ankle/images/')
            cv2.imwrite(newImagePath + 'ankle/images/' + ankleImageName, ankleImg)

            x1List.append(x1)
            x2List.append(x2)
            x3List.append(x3)
            x4List.append(x4)
            x5List.append(x5)
            x6List.append(x6)
            x7List.append(x7)
            x8List.append(x8)
            x9List.append(x9)
            x10List.append(x10)
            x11List.append(x11)
            x12List.append(x12)
            x13List.append(x13)
            x14List.append(x14)

            y1List.append(y1)
            y2List.append(y2)
            y3List.append(y3)
            y4List.append(y4)
            y5List.append(y5)
            y6List.append(y6)
            y7List.append(y7)
            y8List.append(y8)
            y9List.append(y9)
            y10List.append(y10)
            y11List.append(y11)
            y12List.append(y12)
            y13List.append(y13)
            y14List.append(y14)

            hipImageNameList.append(hipImageName)
            kneeImageNameList.append(kneeImageName)
            ankleImageNameList.append(ankleImageName)
        else:
            os.remove(imgPath)
    hipData = {'imgName': hipImageNameList, 'x1': x1List, 'y1': y1List, 'x2': x2List, 'y2': y2List, 'x3': x3List,
               'y3': y3List, 'x4': x4List, 'y4': y4List, 'x5': x5List, 'y5': y5List, 'x6': x6List, 'y6': y6List}
    hipDf = pd.DataFrame(hipData)
    hipDf.to_csv(newImagePath + 'hip/' + 'annotation.csv', index=False)

    kneeData = {'imgName': kneeImageNameList, 'x1': x7List, 'y1': y7List, 'x2': x8List, 'y2': y8List, 'x3': x9List,
                'y3': y9List, 'x4': x10List, 'y4': y10List}
    kneeDf = pd.DataFrame(kneeData)
    kneeDf.to_csv(newImagePath + 'knee/' + 'annotation.csv', index=False)

    ankleData = {'imgName': ankleImageNameList, 'x1': x11List, 'y1': y11List, 'x2': x12List, 'y2': y12List,
                 'x3': x13List,
                 'y3': y13List, 'x4': x14List, 'y4': y14List}
    ankleDf = pd.DataFrame(ankleData)
    ankleDf.to_csv(newImagePath + 'ankle/' + 'annotation.csv', index=False)




def generateTxtAnnotation2(imageFolderPath, newImagePath):
    hipImageNameList, kneeImageNameList, ankleImageNameList = [], [], []
    for imageName in os.listdir(imageFolderPath):
        imagePath = imageFolderPath + imageName
        img = cv2.imread(imagePath)
        h, w, c = img.shape
        third = int(math.floor(h / 3))
        hipImg = img[0:third, :, :]
        kneeImg = img[third:2 * third, :, :]
        ankleImg = img[2 * third:, :, :]

        # save new image and new annotations
        hipImageName = imageName[:-4] + 'hip' + '.jpg'
        if not os.path.exists(newImagePath + 'hip/'):
            os.makedirs(newImagePath + 'hip/')
        cv2.imwrite(newImagePath + 'hip/' + hipImageName, hipImg)

        kneeImageName = imageName[:-4] + 'knee' + '.jpg'
        if not os.path.exists(newImagePath + 'knee/'):
            os.makedirs(newImagePath + 'knee/')
        cv2.imwrite(newImagePath + 'knee/' + kneeImageName, kneeImg)

        ankleImageName = imageName[:-4] + 'ankle' + '.jpg'
        if not os.path.exists(newImagePath + 'ankle/'):
            os.makedirs(newImagePath + 'ankle/')
        cv2.imwrite(newImagePath + 'ankle/' + ankleImageName, ankleImg)

        hipImageNameList.append(hipImageName)
        kneeImageNameList.append(kneeImageName)
        ankleImageNameList.append(ankleImageName)

    hipData = {'imgName': hipImageNameList,}
    hipDf = pd.DataFrame(hipData)
    hipDf.to_csv(newImagePath + 'hip/' + 'annotation.csv', index=False)

    kneeData = {'imgName': kneeImageNameList}
    kneeDf = pd.DataFrame(kneeData)
    kneeDf.to_csv(newImagePath + 'knee/' + 'annotation.csv', index=False)

    ankleData = {'imgName': ankleImageNameList}
    ankleDf = pd.DataFrame(ankleData)
    ankleDf.to_csv(newImagePath + 'ankle/' + 'annotation.csv', index=False)

generateTxtAnnotation2('','')
