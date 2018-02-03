import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

testIndex = [0, 50, 100]

trainTarget = np.delete(iris.target, testIndex)
trainData = np.delete(iris.data, testIndex, axis = 0)

testTarget = iris.target[testIndex]
testData = iris.data[testIndex]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainData, trainTarget)

print(testTarget)
print(clf.predict(testData))