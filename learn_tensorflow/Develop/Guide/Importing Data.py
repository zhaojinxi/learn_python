import tensorflow
import sklearn.preprocessing

(train_data,train_label),(test_data,test_label)=tensorflow.keras.datasets.mnist.load_data()
(train_data,train_label),(test_data,test_label)=tensorflow.keras.datasets.mnist.load_data()
OneHotEncoder=sklearn.preprocessing.OneHotEncoder()
OneHotEncoder.fit(train_label.reshape(-1,1))
train_label=OneHotEncoder.transform(train_label.reshape(-1,1)).toarray()
test_label=OneHotEncoder.transform(test_label.reshape(-1,1)).toarray()

#one-shot
train_dataset=tensorflow.data.Dataset.from_tensor_slices({'data':train_data, 'label':train_label})
test_dataset=tensorflow.data.Dataset.from_tensor_slices({'data':test_data, 'label':test_label})
train_iterator=train_dataset.make_one_shot_iterator()
test_iterator=test_dataset.make_one_shot_iterator()
train_next_element=train_iterator.get_next()
test_next_element=test_iterator.get_next()
Session=tensorflow.Session()
for i in range(3):
    print(Session.run(train_next_element))
    print(Session.run(test_next_element))
Session.close()

#initializable
train_data_placeholder = tensorflow.placeholder(train_data.dtype, train_data.shape)
train_label_placeholder = tensorflow.placeholder(train_label.dtype, train_label.shape)
test_data_placeholder = tensorflow.placeholder(test_data.dtype, test_data.shape)
test_label_placeholder = tensorflow.placeholder(test_label.dtype, test_label.shape)
train_dataset = tensorflow.data.Dataset.from_tensor_slices({'data':train_data_placeholder, 'label':train_label_placeholder})
test_dataset = tensorflow.data.Dataset.from_tensor_slices({'data':test_data_placeholder, 'label':test_label_placeholder})
train_iterator=train_dataset.make_initializable_iterator()
test_iterator=test_dataset.make_initializable_iterator()
train_next_element=train_iterator.get_next()
test_next_element=test_iterator.get_next()
Session=tensorflow.Session()
Session.run(train_iterator.initializer, feed_dict={train_data_placeholder: train_data,train_label_placeholder:train_label})
Session.run(test_iterator.initializer, feed_dict={test_data_placeholder: test_data,test_label_placeholder:test_label})
for i in range(3):
    print(Session.run(train_next_element))
    print(Session.run(test_next_element))
Session.close()

#reinitializable
train_data_placeholder = tensorflow.placeholder(train_data.dtype, train_data.shape)
train_label_placeholder = tensorflow.placeholder(train_label.dtype, train_label.shape)
test_data_placeholder = tensorflow.placeholder(test_data.dtype, test_data.shape)
test_label_placeholder = tensorflow.placeholder(test_label.dtype, test_label.shape)
train_dataset = tensorflow.data.Dataset.from_tensor_slices({'data':train_data_placeholder, 'label':train_label_placeholder})
test_dataset = tensorflow.data.Dataset.from_tensor_slices({'data':test_data_placeholder, 'label':test_label_placeholder})
iterator = tensorflow.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
next_element = iterator.get_next()
train_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)
Session=tensorflow.Session()
Session.run(train_init_op,feed_dict={train_data_placeholder: train_data,train_label_placeholder:train_label})
for i in range(3):
    print(Session.run(next_element))
Session.run(test_init_op,feed_dict={test_data_placeholder: test_data,test_label_placeholder:test_label})
for i in range(3):
    print(Session.run(next_element))
Session.close()

#feedable
train_data_placeholder = tensorflow.placeholder(train_data.dtype, train_data.shape)
train_label_placeholder = tensorflow.placeholder(train_label.dtype, train_label.shape)
test_data_placeholder = tensorflow.placeholder(test_data.dtype, test_data.shape)
test_label_placeholder = tensorflow.placeholder(test_label.dtype, test_label.shape)
train_dataset = tensorflow.data.Dataset.from_tensor_slices({'data':train_data_placeholder, 'label':train_label_placeholder})
test_dataset = tensorflow.data.Dataset.from_tensor_slices({'data':test_data_placeholder, 'label':test_label_placeholder})
handle = tensorflow.placeholder(tensorflow.string, shape=[])
iterator = tensorflow.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
next_element = iterator.get_next()
train_iterator = train_dataset.make_initializable_iterator()
test_iterator = test_dataset.make_initializable_iterator()
Session=tensorflow.Session()
train_handle = Session.run(train_iterator.string_handle())
test_handle = Session.run(test_iterator.string_handle())
Session.run(train_iterator.initializer, feed_dict={train_data_placeholder: train_data,train_label_placeholder:train_label})
Session.run(test_iterator.initializer, feed_dict={test_data_placeholder: test_data,test_label_placeholder:test_label})
for _ in range(3):
    Session.run(next_element, feed_dict={handle: train_handle})
for _ in range(3):
    Session.run(next_element, feed_dict={handle: test_handle})

#Saving iterator state
# Create saveable object from iterator.
saveable = tensorflow.contrib.data.make_saveable_from_iterator(iterator)
# Save the iterator state by adding it to the saveable objects collection.
tensorflow.add_to_collection(tensorflow.GraphKeys.SAVEABLE_OBJECTS, saveable)
saver = tensorflow.train.Saver()
with tensorflow.Session() as sess:
  if should_checkpoint:
    saver.save(path_to_checkpoint)
# Restore the iterator state.
with tensorflow.Session() as sess:
  saver.restore(sess, path_to_checkpoint)