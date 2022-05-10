import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from skimage.feature import hog
import warnings
warnings.filterwarnings("ignore")


data_directory = "Data"

columns=[]
for ctr in range(0,3780):
    columns.append(str(ctr))

categories = ["CatsTrain","DogsTrain","CatsTest","DogsTest"]

def create_data(category):
    path = os.path.join(data_directory,category)
    df = pd.DataFrame (columns=columns)
    for image in os.listdir(path):
        img = cv2.imread(os.path.join(path,image),cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img,(128,64))
        fd,hog_img = hog(resized_img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True)
        df.loc[len(df)] = fd.tolist()
    return df

catsTrain = create_data(categories[0])
catsTrain['label'] = "cat"

dogsTrain = create_data(categories[1])
dogsTrain['label'] = "dog"

catsTest = create_data(categories[2])
catsTest['label'] = "cat"

dogsTest = create_data(categories[3])
dogsTest['label'] = "dog"

print(catsTrain.head())
print(catsTrain.shape)

print(dogsTrain.head())
print(dogsTrain.shape)

print(catsTest.head())
print(catsTest.shape)

print(dogsTest.head())
print(dogsTest.shape)

trainingSet = pd.DataFrame (columns=columns)
testingSet = pd.DataFrame (columns=columns)

trainingSet = pd.concat([catsTrain,dogsTrain],axis=0)
testingSet = pd.concat([catsTest,dogsTest],axis=0)

print(trainingSet.head())

trainingSet['label'] = [1 if animal == 'cat' else 0 for animal in trainingSet['label']]
testingSet['label'] = [1 if animal == 'cat' else 0 for animal in testingSet['label']]

trainingSet = trainingSet.sample(frac=1,random_state = 42)
testingSet = testingSet.sample(frac=1,random_state = 42)

print(trainingSet.head())

print(trainingSet.columns)

from scipy import stats
for column in trainingSet.columns:
    if column != 'label':
        corr,pvalue = stats.pearsonr(trainingSet[column],trainingSet['label'])
        if pvalue >=0.05:
            trainingSet.drop(columns=[column],axis = 1,inplace=True)
            testingSet.drop(columns=[column],axis = 1,inplace=True) 

print(trainingSet.shape)

print(testingSet.shape)

from sklearn import svm
from sklearn.metrics import accuracy_score

y_train = trainingSet['label']
x_train = trainingSet.drop(columns=['label'],axis=1)

y_test = testingSet['label']
x_test = testingSet.drop(columns=['label'],axis=1)

svc = svm.SVC(kernel='linear').fit(x_train, y_train)
print(accuracy_score(svc.predict(x_train),y_train))
print(accuracy_score(svc.predict(x_test),y_test))

lin_svc = svm.LinearSVC().fit(x_train, y_train)
print(accuracy_score(lin_svc.predict(x_train),y_train))
print(accuracy_score(lin_svc.predict(x_test),y_test))

rbf_svc = svm.SVC(kernel='rbf').fit(x_train, y_train)
print(accuracy_score(rbf_svc.predict(x_train),y_train))
print(accuracy_score(rbf_svc.predict(x_test),y_test))

poly_svc = svm.SVC(kernel='poly').fit(x_train, y_train)
print(accuracy_score(poly_svc.predict(x_train),y_train))
print(accuracy_score(poly_svc.predict(x_test),y_test))

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

supportVectorMachines = [("polyKernel",svm.SVC(kernel='poly')),("linearKernel",svm.SVC(kernel='linear')),("rbfKernel",svm.SVC(kernel='rbf')),("linear",svm.LinearSVC())]
classifier = StackingClassifier(estimators=supportVectorMachines,final_estimator=LogisticRegression(),cv=10).fit(x_train,y_train)
print(accuracy_score(classifier.predict(x_train),y_train))
print(accuracy_score(classifier.predict(x_test),y_test))

from sklearn.feature_selection import SelectKBest, chi2

trainingSet = pd.DataFrame (columns=columns)
testingSet = pd.DataFrame (columns=columns)

trainingSet = pd.concat([catsTrain,dogsTrain],axis=0)
testingSet = pd.concat([catsTest,dogsTest],axis=0)

trainingSet['label'] = [1 if animal == 'cat' else 0 for animal in trainingSet['label']]
testingSet['label'] = [1 if animal == 'cat' else 0 for animal in testingSet['label']]

trainingSet = trainingSet.sample(frac=1,random_state = 42)
testingSet = testingSet.sample(frac=1,random_state = 42)

selector = SelectKBest(chi2)

x_train = trainingSet.drop(columns=['label'],axis=1)
y_train = trainingSet['label']

selected_features = selector.fit_transform(x_train,y_train)

print(selected_features.shape)

featureDiscarder = selector.get_support()

features = np.array(columns)

winningFeatures = features[featureDiscarder]

print(winningFeatures)

for column in trainingSet:
    if column != 'label' :
        if column not in winningFeatures:
            trainingSet.drop(columns=[column],axis=1,inplace=True)
            testingSet.drop(columns=[column],axis=1,inplace=True)

y_train = trainingSet['label']
x_train = trainingSet.drop(columns=['label'],axis=1)

y_test = testingSet['label']
x_test = testingSet.drop(columns=['label'],axis=1)

svc = svm.SVC(kernel='linear').fit(x_train, y_train)
lin_svc = svm.LinearSVC().fit(x_train, y_train)
rbf_svc = svm.SVC(kernel='rbf').fit(x_train, y_train)
poly_svc = svm.SVC(kernel='poly').fit(x_train, y_train)

print(accuracy_score(svc.predict(x_test),y_test))
print(accuracy_score(lin_svc.predict(x_test),y_test))
print(accuracy_score(rbf_svc.predict(x_test),y_test))
print(accuracy_score(poly_svc.predict(x_test),y_test))

training_data =[]
def create_training_data():
    for category in categories:
        path = os.path.join(data_directory,category)
        class_index = categories.index(category)
        if class_index== 0 or class_index==2:
            class_index = 1
        else :
            class_index=0
        for images in os.listdir(path):
            try :
                img_array= cv2.imread(os.path.join(path,images),cv2.IMREAD_GRAYSCALE)
                new_image = cv2.resize(img_array,(128,64))
                training_data.append([new_image,class_index])
            except Exception as e:
                pass

create_training_data()

import random
random.shuffle(training_data)

X=[]
Y=[]
for features, labels in training_data:
    X.append(features)
    Y.append(labels)

X= np.array(X).reshape(-1,128,64,1)
Y = np.array(Y).reshape(2200,1)

print(X.shape)

print(Y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.1, random_state=42,stratify=Y)

x_train = x_train/255
x_test=x_test/255

import tensorflow as tf
import keras

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation = 'relu',input_shape=(128,64,1)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(64,activation='relu'),
                                  tf.keras.layers.Dense(8,activation='relu'),
                                   tf.keras.layers.Dense(1,activation='sigmoid')])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01,momentum=0.8),loss = tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=50,verbose=1)

model.evaluate(x_test,y_test)




