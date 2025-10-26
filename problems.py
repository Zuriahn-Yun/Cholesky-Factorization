import cholesky
import numpy as np
import random

# 1 ---------------------------------------------------------
# Generate SPD Pascal Matrix
def create_spd_pascal(p,n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                A[i,j] = p
            elif j == 0:
                A[i,j] = p 
            else:
                A[i,j] = A[i,j-1] + A[i-1,j]
    return A

# Generate Determinant of SPD Pascal
def determinant_spd_pascal(A):
    curr = 1
    L = cholesky.cholesky(A)
    L = L * L.T
    for i in range(len(L)):
        curr  = curr * L[i,i]
    return curr

# 2 ---------------------------------------------------------
def inverse_spd_pascal(A):
    B = 1/2 * (A + A.T)
    inv_B = cholesky.cholesky_inv(B.copy())
    # Preseve Symmetry
    return inv_B
# 3 ---------------------------------------------------------
# B.T * B * x = B.T * b
def least_squares(r):
    B = np.array([[0,2,0],
                  [r,r,0],
                  [r,0,r],
                  [0,1,1]])
    
    b = np.array([2,2 * r,2 * r,2])
    # Reshape to column array
    b = b.reshape(-1,1)
    BT_B = np.matmul(B.T,B)
    BT_b = np.matmul(B.T,b)
    # BT_B is SPD so we can use Cholesky Factor
    # B.T * B* x = B.T * b -> LL.Tx = B.T * b
    # Therefore Ly = B.T * b using forward substitution
    y = cholesky.cholesky_forward_sub(BT_B,BT_b)
    x = cholesky.cholesky_backward_sub(BT_B,y)
    return x

# Numpy doesnt have an exact Solve with Least Squares Function
# Here is an implementation using QR Factorization
def QR_least_squares(r):
    B = np.array([[0,2,0],
                  [r,r,0],
                  [r,0,r],
                  [0,1,1]])
    b = np.array([2,2 * r,2 * r,2])
    b = b.reshape(-1,1)
    Q,R = np.linalg.qr(B)
    p = Q.T @ b
    x = np.linalg.solve(R,p)
    return x
# 4 ---------------------------------------------------------
# Generate Question 4 Matrix
# corner_cofficient and all_ones_coefficient should be set to -1 or 1
def question_4(n,corner_coefficient,all_ones_coefficient):
    A = np.zeros((n,n))
    for i in range(n-1):
        A[i,i] = 2
        A[i,i + 1] = 1 * all_ones_coefficient
        A[i+1,i] = 1 * all_ones_coefficient
    A[0, n - 1] = 1 * corner_coefficient
    A[n - 1,0] = 1 * corner_coefficient
    A[n - 1,n - 1] = 2
    return A

if __name__ == "__main__":
    print("#1 ------------------------------------------------------------------------------------------------------------------")
    for i in range(1):
        p = random.randint(1,10)
        n = random.randint(3,8)
        B = create_spd_pascal(p,n)
        print("N X N, SPD Pascal Matrix, p = ", p,", n =", n)
        print(B)
        det_B = determinant_spd_pascal(B)
        print("Determinant of B = ", det_B)
    print("#2 ------------------------------------------------------------------------------------------------------------------")
    B = 1/2 * (B * B.T)
    B_inv = np.linalg.inv(B)
    B_SPD_bool = cholesky.check_spd(B_inv)
    print("Is inv(B) SPD: ", B_SPD_bool)
    print(B)
    
    print("#3 ------------------------------------------------------------------------------------------------------------------")
    for i in range(1):
        exp = random.randint(9,20)
        r = 10 ** exp
        x = least_squares(r)
        print(x)
    print("#4 ------------------------------------------------------------------------------------------------------------------")