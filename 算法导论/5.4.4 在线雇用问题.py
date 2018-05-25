import math

def ON_LINE_MAXIMUM(k,n):
    bestscore = float('-inf')
    for i in range(k):
        if score[i] > bestscore:
            bestscore = score[i]
    for i in range(k,n):
        if score[i] > bestscore:
            return i
    return n

score=[4,7,2,6,9,0,1,3,5,8]
n=len(score)
k=round(n/math.e)
print(ON_LINE_MAXIMUM(k,n))