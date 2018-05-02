def MERGE(A,p,q,r):
    L=A[p:q+1]
    R=A[q+1:r+1]
    L.append(float('inf'))
    R.append(float('inf'))
    i=0
    j=0
    for k in range(p,r+1):
        if L[i]<=R[j]:
            A[k]=L[i]
            i=i+1
        else:
            A[k]=R[j]
            j=j+1

def MERGE_SORT(A,p,r):
    if p<r:
        q=(p+r)//2
        MERGE_SORT(A,p,q)
        MERGE_SORT(A,q+1,r)
        MERGE(A,p,q,r)
    return A

x=[2,44,6,3,6,29,7,0,6,73,3,1,5,57,89,4]
print(MERGE_SORT(x,0,len(x)-1))