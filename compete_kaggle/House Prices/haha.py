import pandas
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.ensemble
import matplotlib.pyplot  
import numpy
import sklearn.datasets
import sklearn.metrics

#波士顿房价数据  
boston=sklearn.datasets.load_boston()  
x=boston.data  
y=boston.target  

# 随机挑选  
train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = sklearn.model_selection.train_test_split(x, y, train_size=0.8, random_state=33)  

#数据标准化  
ss_x = sklearn.preprocessing.StandardScaler()  
train_x_disorder = ss_x.fit_transform(train_x_disorder)  
test_x_disorder = ss_x.transform(test_x_disorder)  
  
ss_y = sklearn.preprocessing.StandardScaler()  
train_y_disorder = ss_y.fit_transform(train_y_disorder.reshape(-1, 1))  
test_y_disorder=ss_y.transform(test_y_disorder.reshape(-1, 1))  
  
# 多层感知器-回归模型  
model_mlp = sklearn.neural_network.MLPRegressor(solver='adam', hidden_layer_sizes=(20, 20, 20), random_state=1)  
model_mlp.fit(train_x_disorder,train_y_disorder.ravel())  
mlp_score=sklearn.metrics.mean_squared_error(model_mlp.predict(test_x_disorder), test_y_disorder)
print('sklearn多层感知器-回归模型得分',mlp_score)  
  
model_gbr_disorder=sklearn.ensemble.GradientBoostingRegressor()  
model_gbr_disorder.fit(train_x_disorder,train_y_disorder.ravel())  
gbr_score_disorder=sklearn.metrics.mean_squared_error(model_gbr_disorder.predict(test_x_disorder),test_y_disorder)
print('sklearn集成-回归模型得分',gbr_score_disorder)

##网格调参 
#model_gbr_GridSearch=sklearn.ensemble.GradientBoostingRegressor() 
#param_grid = {'n_estimators':range(20,81,10), 
#              'learning_rate': [0.2,0.1, 0.05, 0.02, 0.01 ], 
#              'max_depth': [4, 6,8], 
#              'min_samples_leaf': [3, 5, 9, 14], 
#              'max_features': [0.8,0.5,0.3, 0.1]} 
#estimator = sklearn.model_selection.GridSearchCV(model_gbr_GridSearch,param_grid ) 
#estimator.fit(train_x_disorder,train_y_disorder.ravel() ) 
#print('最优调参：',estimator.best_params_) 
#print('调参后得分',estimator.score(test_x_disorder, test_y_disorder.ravel())) 

#画图
model_gbr_best=sklearn.ensemble.GradientBoostingRegressor(learning_rate=0.1,max_depth=6,max_features=0.5,min_samples_leaf=14,n_estimators=70)  
model_gbr_best.fit(train_x_disorder,train_y_disorder.ravel() )  
gbr_pridict_disorder=model_gbr_disorder.predict(test_x_disorder)  
mlp_pridict_disorder=model_mlp.predict(test_x_disorder)  
  
fig = matplotlib.pyplot.figure(figsize=(20, 3))
axes = fig.add_subplot(1, 1, 1)  
line3,=axes.plot(range(len(test_y_disorder)), test_y_disorder, 'g',label='实际')  
line1,=axes.plot(range(len(gbr_pridict_disorder)), gbr_pridict_disorder, 'b--',label='集成模型',linewidth=2)  
line2,=axes.plot(range(len(mlp_pridict_disorder)), mlp_pridict_disorder, 'r--',label='多层感知器',linewidth=2)  
axes.grid()  
fig.tight_layout()  
matplotlib.pyplot.legend(handles=[line1, line2,line3])   
matplotlib.pyplot.title('sklearn 回归模型')  
matplotlib.pyplot.show()  