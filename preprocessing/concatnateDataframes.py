import pandas as pd

# csvPath1 = '/Users/shiyan/Documents/Mayo/Landmark-Detection/data/annotations/OAIfiltered 1_100.csv'
# csvPath2 = '/Users/shiyan/Documents/Mayo/Landmark-Detection/data/annotations/OAIfiltered 101_200.csv'
# csvPath3 = '/Users/shiyan/Documents/Mayo/Landmark-Detection/data/annotations/OAIfiltered 201_500.csv'
# csvPath4 = '/Users/shiyan/Documents/Mayo/Landmark-Detection/data/annotations/OAIfiltered 501_650.csv'
# csvPath5 = '/Users/shiyan/Documents/Mayo/Landmark-Detection/data/annotations/OAIfiltered 651_800.csv'
# csvPath6 = '/Users/shiyan/Documents/Mayo/Landmark-Detection/data/annotations/OAIfiltered 800_1084.csv'
#
# df1 = pd.read_csv(csvPath1)
# df2 = pd.read_csv(csvPath2)
# df3 = pd.read_csv(csvPath3)
# df4 = pd.read_csv(csvPath4)
# df5 = pd.read_csv(csvPath5)
# df6 = pd.read_csv(csvPath6)
#
# tmpDf = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
# print(tmpDf)
# tmpDf.to_csv('/Users/shiyan/Documents/Mayo/Landmark-Detection/data/annotations/temporary.csv', index=False)

path1 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b1/annotation.csv'
path2 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b2/annotation.csv'
path3 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b3/annotation.csv'
path4 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b4/annotation.csv'
path5 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b5/annotation.csv'
path6 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b6/annotation.csv'
path7 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b7/annotation.csv'
path8 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b8/annotation.csv'
path9 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b9/annotation.csv'
path10 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b10/annotation.csv'
path11 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b11/annotation.csv'
path12 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b12/annotation.csv'
path13 = '/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/b13/annotation.csv'

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)
df4 = pd.read_csv(path4)
df5 = pd.read_csv(path5)
df6 = pd.read_csv(path6)
df7 = pd.read_csv(path7)
df8 = pd.read_csv(path8)
df9 = pd.read_csv(path9)
df10 = pd.read_csv(path10)
df11 = pd.read_csv(path11)
df12 = pd.read_csv(path12)
df13 = pd.read_csv(path13)

tmpDf = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13], axis=0)
tmpDf.to_csv('/infodev1/phi-data/shi/kneeX-ray/data/allOAIdata/temporary.csv', index=False)