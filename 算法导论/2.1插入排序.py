def INSERTION_SORT(A):
    for j in range(1,len(A)):
        key=A[j]
        i=j-1
        while i>=0 and A[i]>key:
            A[i+1]=A[i]
            i=i-1
        A[i+1]=key

    return A

x=[5,3,56,28,4,5,7,3,1,9,7,0]

print(INSERTION_SORT(x))