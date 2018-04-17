import sklearn.ensemble
import sklearn.neighbors
import sklearn.model_selection
import sklearn.datasets
import sklearn.tree
import numpy
import sklearn.metrics
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.svm
import itertools

#1.11.1. Bagging meta-estimator
bagging = sklearn.ensemble.BaggingClassifier(sklearn.neighbors.KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

#1.11.2. Forests of randomized trees
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)

X, y = sklearn.datasets.make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)
clf = sklearn.tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = sklearn.model_selection.cross_val_score(clf, X, y)
scores.mean()                             
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = sklearn.model_selection.cross_val_score(clf, X, y)
scores.mean()                             
clf = sklearn.ensemble.ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = sklearn.model_selection.cross_val_score(clf, X, y)
scores.mean() > 0.999

#1.11.3. AdaBoost
iris = sklearn.datasets.load_iris()
clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
scores = sklearn.model_selection.cross_val_score(clf, iris.data, iris.target)
scores.mean()                             

#1.11.4. Gradient Tree Boosting
X, y = sklearn.datasets.make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]
clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)

X, y = sklearn.datasets.make_friedman1(n_samples=1200, random_state=0, noise=1.0)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]
est = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
sklearn.metrics.mean_squared_error(y_test, est.predict(X_test))    

_ = est.set_params(n_estimators=200, warm_start=True)  # set warm_start and new nr of trees
_ = est.fit(X_train, y_train) # fit additional 100 trees to est
sklearn.metrics.mean_squared_error(y_test, est.predict(X_test))    

X, y = sklearn.datasets.make_hastie_10_2(random_state=0)
clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
clf.feature_importances_  

X, y = sklearn.datasets.make_hastie_10_2(random_state=0)
clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
features = [0, 1, (0, 1)]
fig, axs = sklearn.ensemble.partial_dependence.plot_partial_dependence(clf, X, features) 

iris = sklearn.datasets.load_iris()
mc_clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=10, max_depth=1).fit(iris.data, iris.target)
features = [3, 2, (3, 2)]
fig, axs = sklearn.ensemble.partial_dependence.plot_partial_dependence(mc_clf, X, features, label=0) 

pdp, axes = sklearn.ensemble.partial_dependence.partial_dependence(clf, [0], X=X)
pdp  
axes  

#1.11.5. Voting Classifier
iris = sklearn.datasets.sklearn.datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
clf1 = sklearn.linear_model.LogisticRegression(random_state=1)
clf2 = sklearn.ensemble.RandomForestClassifier(random_state=1)
clf3 = sklearn.naive_bayes.GaussianNB()
eclf = sklearn.ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = sklearn.model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Loading some example data
iris = sklearn.datasets.sklearn.datasets.load_iris()
X = iris.data[:, [0,2]]
y = iris.target
# Training classifiers
clf1 = sklearn.tree.DecisionTreeClassifier(max_depth=4)
clf2 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=7)
clf3 = sklearn.svm.SCV(kernel='rbf', probability=True)
eclf = sklearn.ensemble.VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])
clf1 = clf1.fit(X,y)
clf2 = clf2.fit(X,y)
clf3 = clf3.fit(X,y)
eclf = eclf.fit(X,y)

clf1 = sklearn.linear_model.LogisticRegression(random_state=1)
clf2 = sklearn.ensemble.RandomForestClassifier(random_state=1)
clf3 = sklearn.naive_bayes.GaussianNB()
eclf = sklearn.ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}
grid = sklearn.model_selection.GridSearchCV(estimator=eclf, param_grid=params, cv=5)
grid = grid.fit(iris.data, iris.target)

eclf = sklearn.ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

eclf = sklearn.ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[2,5,1])