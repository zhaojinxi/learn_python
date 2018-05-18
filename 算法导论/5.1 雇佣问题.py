def HIRE_ASSISTANT(n):
    best = 0
    best_qualify=0
    for i in range(len(n)):
        if n[i] > best_qualify:
            best = i
            best_qualify = n[i]
    return  best, best_qualify

n=[4,2,3,5,8,6,7]
print(HIRE_ASSISTANT(n))