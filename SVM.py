import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import scikitplot as skplt


# Opening the files about data
X = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))
Y = pickle.load(open("X_test.pickle","rb"))
y_test = pickle.load(open("y_test.pickle","rb"))

# normalizing data (a pixel goes from 0 to 255) for faster recognition
X = X/255.0
Y = Y/255.0

#reshaping the numpy array from 4D to 2D to fit in SVM classifier
X_train = np.reshape(X, [128,784])
X_test = np.reshape(Y,[40,784])

#training the data
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#predicting the data
y_pred = svclassifier.predict(X_test)

#output metrics 
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#Ploting the confusion matrix with scikitplot library
skplt.metrics.plot_confusion_matrix(y_test,y_pred)
plt.show()
