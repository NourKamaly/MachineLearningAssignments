import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv("Cars_dataset.csv")
print(dataset)
print(dataset.info())
print(np.sum(dataset.isna()))
print(dataset.describe())
print(dataset.corr())
print(np.sum(dataset.duplicated()))
print(dataset.year.value_counts())

high_frequency_year = [2008,2014,2016,2013,2007]
dataset['year']= [x if x in high_frequency_year else 'Other_year' for x in dataset['year']]
print(dataset)

print(dataset.car_maker.value_counts())
high_frequency_car_makers = ['Toyota','Ford','BMW']
dataset['car_maker']= [x if x in high_frequency_car_makers else 'Other_makers' for x in dataset['car_maker']]
print(dataset)

sns.catplot(x = "car_maker", hue = "price", data = dataset, kind = "count")
plt.show()

sns.catplot(x = "year", hue = "price", data = dataset, kind = "count")
plt.show()

sns.catplot(x = "kilometers", hue = "price", data = dataset, kind = "count")
plt.show()

plt.scatter(dataset['kilometers'],dataset['price'])
plt.xlabel('kilometers', fontsize = 10)
plt.ylabel('price', fontsize = 10)
plt.show()

largest_kilometer = dataset['kilometers'].max()
dataset['kilometers'] = [(x/largest_kilometer) for x in dataset['kilometers']]

print(dataset)

print(dataset.describe())

def plot_histogram(dataFrame,column):
    plt.hist(x=dataFrame[column],color='blue',alpha=0.45)
    plt.xlabel(column)
    plt.ylabel('frequency')
    plt.show()

plot_histogram(dataset,'price')

plot_histogram(dataset,'kilometers')

plot_histogram(dataset,'car_maker')

def find_outliers(dataFrame,column):
    Q1= np.percentile(dataFrame[column],25)
    Q3= np.percentile(dataFrame[column],75)
    IQR = Q3-Q1
    floor = Q1 - 1.5*IQR
    ceiling =Q3 +1.5*IQR
    ctr = 0
    outliers=[]
    for observation in dataFrame[column]:
        if observation < floor or observation> ceiling :
            outliers.append(observation)
            ctr+=1
    return floor,ceiling, ctr,outliers

floor,ceiling, ctr,outliers = find_outliers(dataset,'price')
print("lower boundary of price", floor)
print("upper boundary of price",ceiling)
print("number of outliers",ctr)
print("outliers value",outliers)

floor,ceiling, ctr,outliers = find_outliers(dataset,'kilometers')
print("lower boundary of kilometers", floor)
print("upper boundary of kilometers",ceiling)
print("number of outliers",ctr)
print("outliers value",outliers)

print(dataset)

toDummyColumns=['car_maker','year']
for column in toDummyColumns:
    sparse_columns = pd.get_dummies(dataset[column])
    dataset = dataset.drop(column,axis= 1)
    dataset = pd.concat([dataset,sparse_columns],1)

dataset = dataset.sample(frac=1,random_state = 26)

print(dataset)

x = dataset.drop('price',1)
print(x)

y = dataset['price']
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 404)

degrees= []
training_error=[]
testing_error = []

def polynomial_model(X_train, X_test, y_train, y_test,complexity):
    degrees.append(complexity)
    model = Lasso()
    polynomial_features = PolynomialFeatures(degree=complexity)
    expanded_features = polynomial_features.fit_transform(X_train)
    model.fit(expanded_features, y_train)
    trainPrediction = model.predict(expanded_features)
    testPrediction = model.predict(polynomial_features.fit_transform(X_test))

    trainError = metrics.mean_squared_error(y_train, trainPrediction)
    testError = metrics.mean_squared_error(y_test, testPrediction)
    training_error.append(trainError)
    testing_error.append(testError)
    print('Train subset (MSE) for degree {}: '.format(complexity), trainError)
    print('Test subset (MSE) for degree {}: '.format(complexity), testError)

polynomial_model(x_train,x_test,y_train,y_test,4)

polynomial_model(x_train,x_test,y_train,y_test,5)

polynomial_model(x_train,x_test,y_train,y_test,6)

polynomial_model(x_train,x_test,y_train,y_test,7)

polynomial_model(x_train,x_test,y_train,y_test,8)

polynomial_model(x_train,x_test,y_train,y_test,9)

polynomial_model(x_train,x_test,y_train,y_test,10)

polynomial_model(x_train,x_test,y_train,y_test,11)

plt.plot(degrees,training_error,color = 'red',label = 'Train error')
plt.plot(degrees,testing_error,color = 'green',label = 'Test error')
plt.legend(loc="upper left", title="Errors",frameon = False)
plt.show()

plt.plot(degrees,testing_error,color = 'green',label = 'Test error')
plt.xlabel('complexity')
plt.ylabel('testing error')
plt.show()

plt.plot(degrees,training_error,color = 'red',label = 'Train error')
plt.xlabel('complexity')
plt.ylabel('training error')
plt.show()