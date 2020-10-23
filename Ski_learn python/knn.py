import numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')
# print(data.head())


X = data[[
    'buying',
    'maint',
    'safety'
]].values

y = data[['class']]
# print(X,y)

Le = LabelEncoder()
# For converting X into numerical value
for i in range(len(X[0])):
    X[:,i] = Le.fit_transform(X[:,i])

# print(X)
# Same for y
label_mapping = {
    'unacc': 0 ,
    'acc': 1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)
# print(y)

#Creating KNN object model

knn = neighbors.KNeighborsClassifier(n_neighbors=23, weights='uniform')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

knn.fit(X_train,y_train)

prediction= knn.predict(X_test)

acc = metrics.accuracy_score(y_test,prediction)

print("Prediction",prediction)
print("Accuracy", acc)

print("actual value", y[1727])
print("predicted value:",knn.predict(X)[1727])
