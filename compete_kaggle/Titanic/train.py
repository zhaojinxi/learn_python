import pandas
import numpy
import tensorflow
import sklearn.preprocessing
import sys
import os

# read data
train=pandas.read_csv('data/Titanic/train.csv')
test=pandas.read_csv('data/Titanic/test.csv')

# preprocess data
dif_sex=train['Sex'].unique()
sex=train['Sex']
le=sklearn.preprocessing.LabelEncoder()
le.fit(sex)
sex_encoder=le.transform(sex)
train.loc[:,'Sex']=sex_encoder.reshape(-1,1)
dif_embarked=train['Embarked'].unique()
print(dif_embarked)


train_data=train['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_label=train['Survived']