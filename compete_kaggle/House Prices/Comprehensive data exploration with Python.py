import pandas
import matplotlib.pyplot
import seaborn
import numpy
import sklearn.preprocessing
import scipy
import warnings

warnings.filterwarnings('ignore')

#bring in the six packs
df_train = pandas.read_csv('../data/House Prices/train.csv')

#check the decoration
df_train.columns

#descriptive statistics summary
df_train['SalePrice'].describe()

#histogram
seaborn.distplot(df_train['SalePrice']);

#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pandas.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pandas.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#box plot overallqual/saleprice
var = 'OverallQual'
data = pandas.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = matplotlib.pyplot.subplots(figsize=(8, 6))
fig = seaborn.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

var = 'YearBuilt'
data = pandas.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = matplotlib.pyplot.subplots(figsize=(16, 8))
fig = seaborn.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
matplotlib.pyplot.xticks(rotation=90);

#correlation matrix
corrmat = df_train.corr()
f, ax = matplotlib.pyplot.subplots(figsize=(12, 9))
seaborn.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = numpy.corrcoef(df_train[cols].values.T)
seaborn.set(font_scale=1.25)
hm = seaborn.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
matplotlib.pyplot.show()

#scatterplot
seaborn.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
seaborn.pairplot(df_train[cols], size = 2.5)
matplotlib.pyplot.show();

#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pandas.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...

#standardizing data
saleprice_scaled = sklearn.preprocessing.StandardScaler().fit_transform(df_train['SalePrice'][:,numpy.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pandas.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pandas.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#histogram and normal probability plot
seaborn.distplot(df_train['SalePrice'], fit=scipy.stats.norm);
fig = matplotlib.pyplot.figure()
res = scipy.stats.probplot(df_train['SalePrice'], plot=plt)

#applying log transformation
df_train['SalePrice'] = numpy.log(df_train['SalePrice'])

#transformed histogram and normal probability plot
seaborn.distplot(df_train['SalePrice'], fit=scipy.stats.norm);
fig = matplotlib.pyplot.figure()
res = scipy.stats.probplot(df_train['SalePrice'], plot=plt)

#histogram and normal probability plot
seaborn.distplot(df_train['GrLivArea'], fit=scipy.stats.norm);
fig = matplotlib.pyplot.figure()
res = scipy.stats.probplot(df_train['GrLivArea'], plot=plt)

#data transformation
df_train['GrLivArea'] = numpy.log(df_train['GrLivArea'])

#transformed histogram and normal probability plot
seaborn.distplot(df_train['GrLivArea'], fit=scipy.stats.norm);
fig = matplotlib.pyplot.figure()
res = scipy.stats.probplot(df_train['GrLivArea'], plot=plt)

#histogram and normal probability plot
seaborn.distplot(df_train['TotalBsmtSF'], fit=scipy.stats.norm);
fig = matplotlib.pyplot.figure()
res = scipy.stats.probplot(df_train['TotalBsmtSF'], plot=plt)

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pandas.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = numpy.log(df_train['TotalBsmtSF'])

#histogram and normal probability plot
seaborn.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=scipy.stats.norm);
fig = matplotlib.pyplot.figure()
res = scipy.stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

#scatter plot
matplotlib.pyplot.scatter(df_train['GrLivArea'], df_train['SalePrice']);

#scatter plot
matplotlib.pyplot.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);

#convert categorical variable into dummy
df_train = pandas.get_dummies(df_train)