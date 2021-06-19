import csv
import pandas as pd
from collections import Counter
import numpy as np



def readDataFFile():
    trainData = pd.read_csv('TrainData.txt', header=None, delimiter=",")
    testData = pd.read_csv('TestData.txt', header=None, delimiter=",")
    XTrain = trainData.iloc[:, :-1]
    YTrain = trainData.iloc[:, -1]
    XTest = testData.iloc[:, :-1]
    YTest = testData.iloc[:, -1]
    res = [XTrain, YTrain, XTest, YTest]
    return res


def normallizeData(DataX1):
    DataX1 = (DataX1 - DataX1.min()) / (DataX1.max() - DataX1.min())

    return DataX1


def KNNModel(k,Datas):
    XTrain = normallizeData(Datas[0])
    XTest = normallizeData(Datas[2])
    resOfEachtest = []
    for i in range(len(XTest)):
        NXTest = pd.concat([XTest[i:i + 1]] * len(XTrain), ignore_index=True)
        Xminus = NXTest - XTrain
        TestIRes = Xminus * Xminus
        TestIRes["sum"] = TestIRes.sum(axis=1)
        TestIRes['dis'] = np.sqrt(TestIRes['sum'])
        TestIRes['YTrain'] = Datas[1]

        TestIRes1 = pd.DataFrame(TestIRes)

        sortedTestIRes = TestIRes1.sort_values('dis')

        TrainRes = sortedTestIRes[0:k]

        YTrain_List = TrainRes["YTrain"].tolist()
        c = Counter(YTrain_List)
        TheRepeatedYtrain=c.most_common(1)
        resOfEachtest.append(TheRepeatedYtrain[0][0])
    return resOfEachtest


def Get_Accuracy(resOfEachtest,Datas,k):
    Res = pd.DataFrame(resOfEachtest)
    YTEST = pd.DataFrame(Datas[3])

    YTEST.columns = [0]

    Comparison = np.where(Res == YTEST, 1, 0)

    Comparison = pd.DataFrame(Comparison)

    Comparison.columns = ['a']
    Total = Comparison['a'].sum()
    print('K =', k)
    print('Number of correctly classified instances:', Total, 'Total number of instances :', len(Comparison))
    Accuracy = Total / len(Comparison)
    print('Accuracy :', Accuracy)





def Model(k):
    Datas = readDataFFile()
    TrainPredictedOutputs=KNNModel(k,Datas)
    Get_Accuracy(TrainPredictedOutputs,Datas,k)


Model(1)
Model(3)
Model(5)
Model(7)
Model(9)
