import sklearn.tree
import sklearn.datasets
import graphviz

#1.10.1. Classification
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = sklearn.tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

clf.predict([[2., 2.]])

clf.predict_proba([[2., 2.]])

iris = sklearn.datasets.load_iris()
clf = sklearn.tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

dot_data = sklearn.tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 

dot_data = sklearn.tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

clf.predict(iris.data[:1, :])

clf.predict_proba(iris.data[:1, :])

#1.10.2. Regression
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = sklearn.tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])

#1.10.3. Multi-output problems

#1.10.4. Complexity

#1.10.5. Tips on practical use

#1.10.6. Tree algorithms: ID3, C4.5, C5.0 and CART

#1.10.7. Mathematical formulation