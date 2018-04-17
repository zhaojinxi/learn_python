import mxnet
import numpy

mxnet.random.seed(1)
ctx = mxnet.gpu(0)

with open("../../data/nlp/timemachine.txt") as f:
    time_machine = f.read()

print(time_machine[0:500])

print(time_machine[-38075:-37500])
time_machine = time_machine[:-38083]

character_list = list(set(time_machine))
vocab_size = len(character_list)
print(character_list)
print("Length of vocab: %s" % vocab_size)

character_dict = {}
for e, char in enumerate(character_list):
    character_dict[char] = e
print(character_dict)

time_numerical = [character_dict[char] for char in time_machine]

#########################
#  Check that the length is right
#########################
print(len(time_numerical))
#########################
#  Check that the format looks right
#########################
print(time_numerical[:20])
#########################
#  Convert back to text
#########################
print("".join([character_list[idx] for idx in time_numerical[:39]]))

def one_hots(numerical_list, vocab_size=vocab_size):
    result = mxnet.nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result

print(one_hots(time_numerical[:2]))

def textify(embedding):
    result = ""
    indices = mxnet.nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result

print(textify(one_hots(time_numerical[0:40])))

seq_length = 64
# -1 here so we have enough characters for labels later
num_samples = (len(time_numerical) - 1) // seq_length
dataset = one_hots(time_numerical[:seq_length*num_samples]).reshape((num_samples, seq_length, vocab_size))
textify(dataset[0])

batch_size = 32

print('# of sequences in dataset: ', len(dataset))
num_batches = len(dataset) // batch_size
print('# of batches: ', num_batches)
train_data = dataset[:num_batches*batch_size].reshape((batch_size, num_batches, seq_length, vocab_size))
# swap batch_size and seq_length axis to make later access easier
train_data = mxnet.nd.swapaxes(train_data, 0, 1)
train_data = mxnet.nd.swapaxes(train_data, 1, 2)
print('Shape of data set: ', train_data.shape)

for i in range(3):
    print("***Batch %s:***\n %s \n %s \n\n" % (i, textify(train_data[i, :, 0]), textify(train_data[i, :, 1])))
  
labels = one_hots(time_numerical[1:seq_length*num_samples+1])
train_label = labels.reshape((batch_size, num_batches, seq_length, vocab_size))
train_label = mxnet.nd.swapaxes(train_label, 0, 1)
train_label = mxnet.nd.swapaxes(train_label, 1, 2)
print(train_label.shape)

print(textify(train_data[10, :, 3]))
print(textify(train_label[10, :, 3]))

num_inputs = vocab_size
num_hidden = 256
num_outputs = vocab_size
########################
#  Weights connecting the inputs to the hidden layer
########################
Wxh = mxnet.nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
########################
#  Recurrent weights connecting the hidden layer across time steps
########################
Whh = mxnet.nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx) * .01
########################
#  Bias vector for hidden layer
########################
bh = mxnet.nd.random_normal(shape=num_hidden, ctx=ctx) * .01
########################
# Weights to the output nodes
########################
Why = mxnet.nd.random_normal(shape=(num_hidden,num_outputs), ctx=ctx) * .01
by = mxnet.nd.random_normal(shape=num_outputs, ctx=ctx) * .01
# NOTE: to keep notation consistent, we should really use capital letters for hidden layers and outputs, since we are doing batchwise computations]

params = [Wxh, Whh, bh, Why, by]
for param in params:
    param.attach_grad()

def softmax(y_linear, temperature=1.0):
    lin = (y_linear-mxnet.nd.max(y_linear, axis=1).reshape((-1,1))) / temperature # shift each row of y_linear by its max
    exp = mxnet.nd.exp(lin)
    partition =mxnet.nd.sum(exp, axis=1).reshape((-1,1))
    return exp / partition

####################
# With a temperature of 1 (always 1 during training), we get back some set of probabilities
####################
softmax(mxnet.nd.array([[1, -1], [-1, 1]]), temperature=1.0)

####################
# If we set a high temperature, we can get more entropic (*noisier*) probabilities
####################
softmax(mxnet.nd.array([[1,-1],[-1,1]]), temperature=1000.0)

####################
# Often we want to sample with low temperatures to produce sharp probabilities
####################
softmax(mxnet.nd.array([[10,-10],[-10,10]]), temperature=.1)

def simple_rnn(inputs, state, temperature=1.0):
    outputs = []
    h = state
    for X in inputs:
        h_linear = mxnet.nd.dot(X, Wxh) + mxnet.nd.dot(h, Whh) + bh
        h = mxnet.nd.tanh(h_linear)
        yhat_linear = mxnet.nd.dot(h, Why) + by
        yhat = softmax(yhat_linear, temperature=temperature)
        outputs.append(yhat)
    return (outputs, h)

# def cross_entropy(yhat, y):
#     return - mxnet.nd.sum(y * mxnet.nd.log(yhat))
def cross_entropy(yhat, y):
    return - mxnet.nd.mean(mxnet.nd.sum(y * mxnet.nd.log(yhat), axis=0, exclude=True))

cross_entropy(mxnet.nd.array([[.2,.5,.3], [.2,.5,.3]]), mxnet.nd.array([[1.,0,0], [0, 1.,0]]))

def average_ce_loss(outputs, labels):
    assert(len(outputs) == len(labels))
    total_loss = 0.
    for (output, label) in zip(outputs,labels):
        total_loss = total_loss + cross_entropy(output, label)
    return total_loss / len(outputs)

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def sample(prefix, num_chars, temperature=1.0):
    #####################################
    # Initialize the string that we'll return to the supplied prefix
    #####################################
    string = prefix

    #####################################
    # Prepare the prefix as a sequence of one-hots for ingestion by RNN
    #####################################
    prefix_numerical = [character_dict[char] for char in prefix]
    input = one_hots(prefix_numerical)

    #####################################
    # Set the initial state of the hidden representation ($h_0$) to the zero vector
    #####################################
    sample_state = mxnet.nd.zeros(shape=(1, num_hidden), ctx=ctx)

    #####################################
    # For num_chars iterations,
    #     1) feed in the current input
    #     2) sample next character from from output distribution
    #     3) add sampled character to the decoded string
    #     4) prepare the sampled character as a one_hot (to be the next input)
    #####################################
    for i in range(num_chars):
        outputs, sample_state = simple_rnn(input, sample_state, temperature=temperature)
        choice = numpy.random.choice(vocab_size, p=outputs[-1][0].asnumpy())
        string += character_list[choice]
        input = one_hots([choice])
    return string

epochs = 2000
moving_loss = 0.
learning_rate = .5
# state = mxnet.nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
for e in range(epochs):
    ############################
    # Attenuate the learning rate by a factor of 2 every 100 epochs.
    ############################
    if ((e+1) % 100 == 0):
        learning_rate = learning_rate / 2.0
    state = mxnet.nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    for i in range(num_batches):
        data_one_hot = train_data[i]
        label_one_hot = train_label[i]
        with mxnet.autograd.record():
            outputs, state = simple_rnn(data_one_hot, state)
            loss = average_ce_loss(outputs, label_one_hot)
            loss.backward()
        SGD(params, learning_rate)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        if (i == 0) and (e == 0):
            moving_loss = numpy.mean(loss.asnumpy()[0])
        else:
            moving_loss = .99 * moving_loss + .01 * numpy.mean(loss.asnumpy()[0])

    print("Epoch %s. Loss: %s" % (e, moving_loss))
    print(sample("The Time Ma", 1024, temperature=.1))
    print(sample("The Medical Man rose, came to the lamp,", 1024, temperature=.1))