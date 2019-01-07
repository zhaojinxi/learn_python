import matplotlib.pyplot
import numpy

class DecayFunction():
    def __init__(self, name, learning_rate=0.1, decay_steps=10, decay_rate=0.1, staircase=False):
        self.name = name
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

class ExponentialDecay(DecayFunction):
    def __init__(self, name='exponential_decay', learning_rate=0.1, decay_steps=50, decay_rate=0.1, staircase=False):
        return super().__init__(name, learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)

    def lr(self, global_step):
        if self.staircase:
            result = self.learning_rate * self.decay_rate ** numpy.floor(global_step / self.decay_steps)
        else:
            result = self.learning_rate * self.decay_rate ** (global_step / self.decay_steps)
        return result

class NaturalExpDecay(DecayFunction):
    def __init__(self, name='natural_exp_decay', learning_rate=0.1, decay_steps=50, decay_rate=1, staircase=False):
        return super().__init__(name, learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)

    def lr(self, global_step):
        if self.staircase:
            result = self.learning_rate * numpy.exp(-self.decay_rate * numpy.floor(global_step / self.decay_steps))
        else:
            result = self.learning_rate * numpy.exp(-self.decay_rate * global_step / self.decay_steps)
        return result

class InverseTimeDecay(DecayFunction):
    def __init__(self, name='inverse_time_decay', learning_rate=0.1, decay_steps=10, decay_rate=0.1, staircase=False):
        return super().__init__(name, learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)

    def lr(self, global_step):      
        if self.staircase:
            result = self.learning_rate / (1 + self.decay_rate * numpy.floor(global_step / self.decay_steps))
        else:
            result = self.learning_rate / (1 + self.decay_rate * global_step / self.decay_steps)
        return result

class LinearCosineDecay(DecayFunction):
    def __init__(self, name='linear_cosine_decay', learning_rate=0.1, decay_steps=50, num_periods=5, alpha=0, beta=0.01,):
        self.num_periods = num_periods
        self.alpha = alpha
        self.beta = beta
        return super().__init__(name, learning_rate=learning_rate, decay_steps=decay_steps)

    def lr(self, global_step):
        global_step = min(global_step, self.decay_steps)
        linear_decay = (self.decay_steps - global_step) / self.decay_steps
        cosine_decay = 0.5 * (1 + numpy.cos(numpy.pi * 2 * self.num_periods * global_step / self.decay_steps))
        decayed = (self.alpha + linear_decay) * cosine_decay + self.beta
        result = self.learning_rate * decayed
        return result

class NoisyLinearCosineDecay(DecayFunction):
    def __init__(self, name='noisy_linear_cosine_decay', learning_rate=0.1, decay_steps=50, initial_variance=1.0, variance_decay=0.55, num_periods=5, alpha=0.0, beta=0.01):
        self.initial_variance = initial_variance
        self.variance_decay = variance_decay
        self.num_periods = num_periods
        self.alpha = alpha
        self.beta = beta
        return super().__init__(name, learning_rate=learning_rate, decay_steps=decay_steps)

    def lr(self, global_step):
        global_step = min(global_step, self.decay_steps)
        linear_decay = (self.decay_steps - global_step) / self.decay_steps
        cosine_decay = 0.5 * (1 + numpy.cos(numpy.pi * 2 * self.num_periods * global_step / self.decay_steps))
        eps_t = numpy.random.normal(0, (self.initial_variance / (1 + global_step) ** self.variance_decay) ** 0.5)
        decayed = (self.alpha + linear_decay + eps_t) * cosine_decay + self.beta
        result = self.learning_rate * decayed
        return result

class CosineDecayRestarts(DecayFunction):
    def __init__(self, name='cosine_decay_restarts', learning_rate=0.1, first_decay_steps=50, t_mul=2.0, m_mul=1.0, alpha=0.0,):
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        return super().__init__(name, learning_rate=learning_rate)

    def lr(self, global_step):
        pass

class CosineDecay(DecayFunction):
    def __init__(self, name='cosine_decay', learning_rate=0.1, decay_steps=50, alpha=0.01):
        self.alpha = alpha
        return super().__init__(name, learning_rate=learning_rate, decay_steps=decay_steps)
    
    def lr(self, global_step):
        global_step = min(global_step, self.decay_steps)
        cosine_decay = 0.5 * (1 + numpy.cos(numpy.pi * global_step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        result = self.learning_rate * decayed
        return result

class PolynomialDecay(DecayFunction):
    def __init__(self, name='polynomial_decay', learning_rate=0.1, decay_steps=50, end_learning_rate=0.01, power=3, cycle=False,):
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        return super().__init__(name, learning_rate=learning_rate, decay_steps=decay_steps)

    def lr(self, global_step):
        if self.cycle:
            self.decay_steps = self.decay_steps * numpy.ceil(global_step / self.decay_steps)
            result = (self.learning_rate - self.end_learning_rate) * (1 - global_step / self.decay_steps) ** (self.power) + self.end_learning_rate
        else:
            global_step = min(global_step, self.decay_steps)
            result = (self.learning_rate - self.end_learning_rate) * (1 - global_step / self.decay_steps) ** (self.power) + self.end_learning_rate
        return result

all_decay=[]
all_decay.append(ExponentialDecay())
all_decay.append(NaturalExpDecay())
all_decay.append(InverseTimeDecay())
all_decay.append(LinearCosineDecay())
all_decay.append(NoisyLinearCosineDecay())
# all_decay.append(CosineDecayRestarts())
all_decay.append(CosineDecay())
all_decay.append(PolynomialDecay())


for one in all_decay:  
    x = list(range(100))
    y = []
    for global_step in x:
        lr = one.lr(global_step=global_step)
        y.append(lr)

    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(x, y)
    ax.set(xlabel='global_step', ylabel='learning_rate', title='%s'%one.name)
    ax.grid()
    fig.savefig("%s.png"%one.name)
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()