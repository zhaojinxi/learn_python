import random
import numpy

def RANDOMIZED_HIRE_ASSISTANT(n):
    random.shuffle(n)
    best = 0
    best_qualify = 0
    for i in range(len(n)):
        if n[i] > best_qualify:
            best = i
            best_qualify = n[i]
    return best, best_qualify

n=[1,2,3,4,5,6,7,8,9]
print(RANDOMIZED_HIRE_ASSISTANT(n))

def PERMUTE_BY_SORTING(A):
    n = len(A)
    P=numpy.zeros(n)
    for i in range (n):
        P[i] = random.randint(1,n**3)
    P=P.argsort()
    A=numpy.array(A)[P]
    return A.tolist()

A=[1,2,3,4,5,6,7]
print(PERMUTE_BY_SORTING(A))

def RANDOMIZE_IN_PLACE(A):
    n = len(A)
    for i in range(n):
        j=random.randint(i,n-1)
        k=A[i]
        A[i]=A[j]
        A[j]=k
    return A

A=[1,2,3,4,5,6,7,8,9]
print(RANDOMIZE_IN_PLACE(A))