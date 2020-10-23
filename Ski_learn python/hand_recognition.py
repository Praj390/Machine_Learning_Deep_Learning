from pil import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist

#Training variables
X_train = mnist.train_images()
y_train = mnist.train_labels()

X_test = mnist.test_images()
y_test = mnist.test_labels()

print('X_train', X_train)
print('X_test',X_test)

print('y_train', y_train)
print('y_test',y_test)

print(X_train.shape)

X_train = X_train.reshape((-1,28*28))
X_test = X_test.reshape((-1,28*28))

print(X_train[0])

X_train = (X_train/256)
X_test = (X_test/256)

clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64*64))

clf.fit(X_train,y_train)

prediction = clf.predict(X_test)
acc = confusion_matrix(y_test,prediction)

print(acc)

def accuracy(cm):
    diagonal = cm.trace()
    element = cm.sum()
    return diagonal/element

print(accuracy(acc))