import torch # This is all you need to use both PyTorch and TorchScript!
print(torch.__version__)

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()

    def forward(self, x, h):
        new_h = torch.tanh(x + h)
        return new_h, new_h
my_cell = MyCell()
x = torch.rand(3, 4)
h = torch.rand(3, 4)
print(my_cell(x, h))

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h
my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))

class MyDecisionGate(torch.nn.Module):
  def forward(self, x):
    if x.sum() > 0:
      return x
    else:
      return -x
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.dg = MyDecisionGate()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h
my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h
my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell)
traced_cell(x, h)

print(traced_cell.graph)

print(traced_cell.code)

print(my_cell(x, h))
print(traced_cell(x, h))

class MyDecisionGate(torch.nn.Module):
  def forward(self, x):
    if x.sum() > 0:
      return x
    else:
      return -x
class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h
my_cell = MyCell(MyDecisionGate())
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell.code)

scripted_gate = torch.jit.script(MyDecisionGate())
my_cell = MyCell(scripted_gate)
traced_cell = torch.jit.script(my_cell)
print(traced_cell.code)

# New inputs
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell(x, h)

class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h
rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)

class WrapRNN(torch.nn.Module):
  def __init__(self):
    super(WrapRNN, self).__init__()
    self.loop = torch.jit.script(MyRNNLoop())

  def forward(self, xs):
    y, h = self.loop(xs)
    return torch.relu(y)
traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
print(traced.code)

traced.save('wrapped_rnn.zip')
loaded = torch.jit.load('wrapped_rnn.zip')
print(loaded)
print(loaded.code)