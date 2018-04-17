1 + 1
a = 4
type(a)   

c = 2.1
type(c)   

a = 1.5 + 0.5j
a.real
a.imag
type(1. + 0j)

3 > 4
test = (3 > 4)
test
type(test)      

7 * 3.
2**10
8 % 3

float(1)

3 / 2

3 / 2

3 / 2.
a = 3
b = 2
a / b # In Python 2  
a / float(b)

from __future__ import division  
3 / 2

3.0 // 2

colors = ['red', 'blue', 'green', 'black', 'white']
type(colors)

colors[2]

colors[-1]
colors[-2]

colors
colors[2:4]

colors
colors[3:]
colors[:3]
colors[::2]

colors[0] = 'yellow'
colors
colors[2:4] = ['gray', 'purple']
colors

colors = [3, -200, 'hello']
colors
colors[1], colors[2]

colors = ['red', 'blue', 'green', 'black', 'white']
colors.append('pink')
colors
colors.pop() # removes and returns the last item
colors
colors.extend(['pink', 'purple']) # extend colors, in-place
colors
colors = colors[:-2]
colors

rcolors = colors[::-1]
rcolors
rcolors2 = list(colors)
rcolors2
rcolors2.reverse() # in-place
rcolors2

rcolors + colors
rcolors * 2

sorted(rcolors) # new object
rcolors
rcolors.sort()  # in-place
rcolors

rcolors.<TAB>

'Hi, what's up?'

a = "hello"
a[0]
a[1]
a[-1]

a = "hello, world!"
a[3:6] # 3rd to 6th (excluded) elements: elements 3, 4, 5
a[2:10:2] # Syntax: a[start:stop:step]
a[::3] # every three characters, from beginning to end

a = "hello, world!"
a[2] = 'z'

a.replace('l', 'z', 1)

a.replace('l', 'z')

'An integer: %i; a float: %f; another string: %s' % (1, 0.1, 'string')
i = 102
filename = 'processing_of_dataset_%d.txt' % i
filename

tel = {'emmanuelle': 5752, 'sebastian': 5578}
tel['francis'] = 5915
tel     
tel['sebastian']
tel.keys()   
tel.values()   
'francis' in tel

d = {'a':1, 'b':2, 3:'hello'}
d

t = 12345, 54321, 'hello!'
t[0]
t
u = (0, 2)

s = set(('a', 'b', 'c', 'a'))
s
s.difference(('a', 'b'))  

a = [1, 2, 3]
b = a
a
b
a is b
b[1] = 'hi!'
a

a = [1, 2, 3]
a
a = ['a', 'b', 'c'] # Creates another object.
a
id(a)
a[:] = [1, 2, 3] # Modifies object in place.
a
id(a)