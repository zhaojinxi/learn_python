import numpy

def Strassen(A,B):
    n=A.shape[0]
    C=numpy.zeros([n,n])
    if n==1:
        C[0][0]=A[0][0]*B[0][0]
    else:
        A11=A[:n//2,:n//2]
        A12=A[:n//2,n//2:]
        A21=A[n//2:,:n//2]
        A22=A[n//2:,n//2:]
        B11=B[:n//2,:n//2]
        B12=B[:n//2,n//2:]
        B21=B[n//2:,:n//2]
        B22=B[n//2:,n//2:]

        S1=B12-B22
        S2=A11+A12
        S3=A21+A22
        S4=B21-B11
        S5=A11+A22
        S6=B11+B22
        S7=A12-A22
        S8=B21+B22
        S9=A11-A21
        S10=B11+B12

        P1=Strassen(A11,S1)
        P2=Strassen(S2,B22)
        P3=Strassen(S3,B11)
        P4=Strassen(A22,S4)
        P5=Strassen(S5,S6)
        P6=Strassen(S7,S8)
        P7=Strassen(S9,S10)

        C11=P5+P4-P2+P6
        C12=P1+P2
        C21=P3+P4
        C22=P5+P1-P3-P7
        C=numpy.concatenate((numpy.concatenate((C11,C12),1),numpy.concatenate((C21,C22),1)),0)
    return C

A=numpy.random.randint(0,10,[4,4])
B=numpy.random.randint(0,10,[4,4])
print(Strassen(A,B))