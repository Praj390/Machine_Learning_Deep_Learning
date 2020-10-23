from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd
from time import time
from sklearn import metrics

bc = load_breast_cancer()

X = scale(bc.data)
print(X)

y = bc.target
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)

model = KMeans(n_clusters=2,random_state=0)

model.fit(X_train)

predictions = model.predict(X_test)

labels = model.labels_

print("Labels",labels)
print("predictions:", predictions)
print("Accuracy:",accuracy_score(y_test,predictions))
print("Actual values:", y_test)

print(pd.crosstab(y_train,labels))

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

bench_k_means(model,'1',X)


