import pandas
import matplotlib.pyplot
import matplotlib.image
import sklearn.model_selection
import sklearn.svm

labeled_images = pandas.read_csv('./data/Digit Recognizer/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = sklearn.model_selection.train_test_split(images, labels, train_size=0.8, random_state=0)

i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
matplotlib.pyplot.imshow(img,cmap='gray')
matplotlib.pyplot.title(train_labels.iloc[i,0])
matplotlib.pyplot.show()
matplotlib.pyplot.hist(train_images.iloc[i])
matplotlib.pyplot.show()

clf = sklearn.svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

test_images[test_images>0]=1
train_images[train_images>0]=1
img=train_images.iloc[i].as_matrix().reshape((28,28))
matplotlib.pyplot.imshow(img,cmap='binary')
matplotlib.pyplot.title(train_labels.iloc[i])
matplotlib.pyplot.show()
matplotlib.pyplot.hist(train_images.iloc[i])
matplotlib.pyplot.show()

clf = sklearn.svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

test_data=pandas.read_csv('./data/Digit Recognizer/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])

results

df = pandas.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)