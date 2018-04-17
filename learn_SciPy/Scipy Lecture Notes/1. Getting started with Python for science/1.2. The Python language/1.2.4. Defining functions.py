def test():
    print('in test function')

test()

def disk_area(radius):
    return 3.14 * radius * radius
disk_area(1.5)

def double_it(x):
    return x * 2
double_it(3)
double_it()

def double_it(x=2):
    return x * 2
double_it()
double_it(3)

bigx = 10
def double_it(x=bigx):
    return x * 2
bigx = 1e9  # Now really big
double_it()

def add_to_dict(args={'a': 1, 'b': 2}):
    for i in args.keys():
        args[i] += 1
    print(args)
add_to_dict
add_to_dict()
add_to_dict()
add_to_dict()

def slicer(seq, start=None, stop=None, step=None):
    """Implement basic python slicing."""
    return seq[start:stop:step]
rhyme = 'one fish, two fish, red fish, blue fish'.split()
rhyme
slicer(rhyme)
slicer(rhyme, step=2)
slicer(rhyme, 1, step=2)
slicer(rhyme, start=1, stop=4, step=2)
slicer(rhyme, step=2, start=1, stop=4)

def try_to_modify(x, y, z):
    x = 23
    y.append(42)
    z = [99] # new reference
    print(x)
    print(y)
    print(z)
a = 77    # immutable variable
b = [99]  # mutable variable
c = [28]
try_to_modify(a, b, c)
print(a)
print(b)
print(c)

x = 5
def addx(y):
    return x + y
addx(10)

def setx(y):
    x = y
    print('x is %d' % x)
setx(10)
x

def setx(y):
    global x
    x = y
    print('x is %d' % x)
setx(10)
x

def variable_args(*args, **kwargs):
    print('args is', args)
    print('kwargs is', kwargs)
variable_args('one', 'two', x=1, y=2, z=3)

def funcname(params):
    """Concise one-line sentence describing the function.

    Extended summary which can contain multiple paragraphs.
    """
    # function body
    pass
funcname?

va = variable_args
va('three', x=1, y=2)