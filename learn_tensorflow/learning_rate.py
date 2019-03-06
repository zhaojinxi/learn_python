import tensorflow
import matplotlib.pyplot

global_step = tensorflow.placeholder(tensorflow.int32, [], 'global_step')

exponential_decay = tensorflow.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=30, decay_rate=0.631, staircase=True, name='exponential_decay')
natural_exp_decay = tensorflow.train.natural_exp_decay(learning_rate=0.1, global_step=global_step, decay_steps=30, decay_rate=0.46, staircase=True, name='natural_exp_decay')
inverse_time_decay = tensorflow.train.inverse_time_decay(learning_rate=0.1, global_step=global_step, decay_steps=33, decay_rate=11, staircase=True, name='inverse_time_decay')
linear_cosine_decay = tensorflow.train.linear_cosine_decay(learning_rate=0.1, global_step=global_step, decay_steps=200, num_periods=3, alpha=0, beta=0.01, name='linear_cosine_decay')
noisy_linear_cosine_decay = tensorflow.train.noisy_linear_cosine_decay(learning_rate=0.1, global_step=global_step, decay_steps=200, initial_variance=0.01, variance_decay=0.55, num_periods=3, alpha=0, beta=0.01, name='noisy_linear_cosine_decay')
cosine_decay = tensorflow.train.cosine_decay(learning_rate=0.1, global_step=global_step, decay_steps=150, alpha=0.01, name='cosine_decay')
cosine_decay_restarts = tensorflow.train.cosine_decay_restarts(learning_rate=0.1, global_step=global_step, first_decay_steps=30, t_mul=3, m_mul=0.5, alpha=0.01, name='cosine_decay_restarts')
polynomial_decay = tensorflow.train.polynomial_decay(learning_rate=0.1, global_step=global_step, decay_steps=100, end_learning_rate=0.001, power=2, cycle=True, name='polynomial_decay')
piecewise_constant_decay = tensorflow.train.piecewise_constant_decay(x=global_step, boundaries=[100, 200], values=[0.1, 0.01, 0.001], name='piecewise_constant_decay')

all_decay = [exponential_decay, natural_exp_decay, inverse_time_decay, linear_cosine_decay, noisy_linear_cosine_decay, cosine_decay, cosine_decay_restarts, polynomial_decay, piecewise_constant_decay]

Session = tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())

for one in all_decay:
    x = list(range(300))
    y = []
    for i in x:
        lr = Session.run(one,feed_dict={global_step:i})
        y.append(lr)

    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(x, y)
    ax.set(xlabel='global_step', ylabel='learning_rate', title='%s'%one.name)
    ax.grid()
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()