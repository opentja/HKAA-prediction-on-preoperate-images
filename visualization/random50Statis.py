import numpy as np
import statistics


def extratAngeDiff(path):
    diffList = []
    al = np.load(path + 'leftAngleLabelList.npy')
    ar = np.load(path + 'rightAngleLabelList.npy')
    apl = np.load(path + 'leftAnglePredList.npy')
    apr = np.load(path + 'rightAnglePredList.npy')
    a = al + ar
    ap = apl + apr
    print('number of images', len(al))
    for i in range(len(al)):
        diff = abs(a[i] - ap[i])
        diffList.append(diff)
    return diffList


a = extratAngeDiff('randomSelect50/ankleToHip0005PredRes/')
b = extratAngeDiff('randomSelect50/ankleToHip0510PredRes/')
c = extratAngeDiff('randomSelect50/ankleToHip1015PredRes/')
d = extratAngeDiff('randomSelect50/ankleToHip1520PredRes/')

diffStd = statistics.stdev(a + b + c + d)
diffMean = statistics.mean(a + b + c + d)

print('diff mean:', diffMean)
print('diff std: ', diffStd)
e = a + b + c + d
count = len([i for i in e if i <= 2])
print(count / len(e))
