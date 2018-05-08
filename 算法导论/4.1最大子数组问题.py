def FIND_MAX_CROSSING_SUBARRAY(A,low,mid,high):
    left_sum=float('-inf')
    sum=0
    for i in range(mid,low-1,-1):
        sum=sum+A[i]
        if sum>left_sum:
            left_sum=sum
            max_left=i
    right_sum=float('-inf')
    sum=0
    for j in range(mid+1,high+1):
        sum=sum+A[j]
        if sum>right_sum:
            right_sum=sum
            max_right=j
    return max_left,max_right,left_sum+right_sum

def FIND_MAXIMUM_SUBARRAY(A,low,high):
    if high==low:
        return low,high,A[low]
    else:
        mid=(low+high)//2
        (left_low,left_high,left_sum)=FIND_MAXIMUM_SUBARRAY(A,low,mid)
        (right_low,right_high,right_sum)=FIND_MAXIMUM_SUBARRAY(A,mid+1,high)
        (cross_low,cross_high,cross_sum)=FIND_MAX_CROSSING_SUBARRAY(A,low,mid,high)
        if left_sum>=right_sum and left_sum>=cross_sum:
            return left_low,left_high,left_sum
        elif right_sum>=left_sum and right_sum>=cross_sum:
            return right_low,right_high,right_sum
        else:
            return cross_low,cross_high,cross_sum

A=[5,-3,7,-1,2,-4,8,-1,-8,-9,0,-3,5,6,-3,5]
print(FIND_MAXIMUM_SUBARRAY(A,0,len(A)-1))