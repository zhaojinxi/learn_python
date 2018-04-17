import tensorflow

#Saving and restoring variables

# Create some variables.
v1 = tensorflow.get_variable("v1", shape=[3], initializer = tensorflow.zeros_initializer)
v2 = tensorflow.get_variable("v2", shape=[5], initializer = tensorflow.zeros_initializer)
inc_v1 = v1.assign(v1 + 1)
dec_v2 = v2.assign(v2 - 1)
# Add an op to initialize the variables.
init_op = tensorflow.global_variables_initializer()
# Add ops to save and restore all the variables.
saver = tensorflow.train.Saver()
# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tensorflow.Session() as sess:
    sess.run(init_op)
    # Do some work with the model.
    inc_v1.op.run()
    dec_v2.op.run()
    # Save the variables to disk.
    save_path = saver.save(sess, "../Model/model.ckpt")
    print("Model saved in file: %s" % save_path)

tensorflow.reset_default_graph()
# Create some variables.
v1 = tensorflow.get_variable("v1", shape=[3])
v2 = tensorflow.get_variable("v2", shape=[5])
# Add ops to save and restore all the variables.
saver = tensorflow.train.Saver()
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tensorflow.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "../Model/model.ckpt")
    print("Model restored.")
    # Check the values of the variables
    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())

tensorflow.reset_default_graph()
# Create some variables.
v1 = tensorflow.get_variable("v1", [3], initializer = tensorflow.zeros_initializer)
v2 = tensorflow.get_variable("v2", [5], initializer = tensorflow.zeros_initializer)
# Add ops to save and restore only `v2` using the name "v2"
saver = tensorflow.train.Saver({"v2": v2})
# Use the saver object normally after that.
with tensorflow.Session() as sess:
  # Initialize v1 since the saver will not.
  v1.initializer.run()
  saver.restore(sess, "../Model/model.ckpt")
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())

#Overview of saving and restoring models

#APIs to build and load a SavedModel

export_dir = "../Model/model.ckpt"
builder = tensorflow.saved_model.builder.SavedModelBuilder(export_dir)
with tensorflow.Session(graph=tensorflow.Graph()) as sess:
    builder.add_meta_graph_and_variables(sess, [tag_constants.TRAINING], signature_def_map=foo_signatures, assets_collection=foo_assets)
# Add a second MetaGraphDef for inference.
with tensorflow.Session(graph=tensorflow.Graph()) as sess:
    builder.add_meta_graph([tag_constants.SERVING])
builder.save()

with tensorflow.Session(graph=tensorflow.Graph()) as sess:
  tensorflow.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)

#Using SavedModel with Estimators

feature_spec = {'foo': tf.FixedLenFeature(...),
                'bar': tf.VarLenFeature(...)}
def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[default_batch_size],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

from grpc.beta import implementations
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = classification_pb2.ClassificationRequest()
example = request.input.example_list.examples.add()
example.features.feature['x'].float_list.value.extend(image[0].astype(float))
result = stub.Classify(request, 10.0)  # 10 secs timeout