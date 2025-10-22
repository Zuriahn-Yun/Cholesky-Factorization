import cholesky
import numpy as np

# 1 
def create_spd_pascal(p,n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                A[i,j] = p
            elif j == 0:
                print("HEre")
                A[i,j] = p 
            else:
                A[i,j] = A[i,j-1] + A[i-1,j]
    return A

def determinant_spd_pascal(A):
    curr = 1
    L = cholesky(A)
    for i in range(L.size[0]):
        curr += curr * L[i,i]
    return curr ** 2

# 2


if __name__ == "__main__":
    A = create_spd_pascal(5,3)
    print(A)