from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()

#Feature/label

X = boston.data
y = boston.target

print("X")
print(X)
print(X.shape)

print("y")
print(y)
print(y.shape)

#Algortihm for linear regression
l_reg = linear_model.LinearRegression()

plt.scatter(X.T[0], y)
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

# Train a model
model = l_reg.fit(X_train,y_train)
prediction = model.predict(X_test)
print("Prediction:", prediction)
print("R^2 value", l_reg.score(X,y))
print("coedd:", l_reg.coef_)
print("intercept:", l_reg.intercept_)