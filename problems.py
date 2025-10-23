import cholesky
import numpy as np

# 1 --------------------------------------------
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

# 2 --------------------------------------------
# the inverse of an SPD is also SPD, compute the iverses B of the Matrices in question 1 --------------- CHECK THIS Is it okay to use built in inverse function
def inverse_spd_pascal(A):
    inv_A = np.linalg.inv(A)
    # Preseve Symmetry
    B = 1/2 * (inv_A + inv_A.T)
    return B

# 3 --------------------------------------------
# any solution x of the least squares problem
# B.T * B * x = B.T * b
# When r is one million this always breaks??--------------------------
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
    x = cholesky.cholesk_backward_sub(BT_B,y)
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
# 4 -------------------------------------------- ask about the contents of the given matrix 
def question_4(n):
    A = np.zeros((n,n))
    for i in range(n-1):
        A[i,i] = 2
        A[i,i + 1] = - 1
        A[i+1,i] = -1
    A[0, n - 1] = 1
    A[n - 1,0] = 1
    A[n - 1,n - 1] = 2
    return A
if __name__ == "__main__":
    # A = create_spd_pascal(15,3)
    # Original_A = A.copy()
    # det_A = determinant_spd_pascal(A)
    # print(A)
    # L = A
    # print(L * L.T)
    # print(det_A)
    
    # B = inverse_spd_pascal(Original_A)
    # print(B)
    # B_cholesky = cholesky.cholesky(B)
    # print(B_cholesky)
    # print(B_cholesky * B_cholesky.T)
    
    # r = 100000
    # x_qr = QR_least_squares(r)
    # x_cholesky = least_squares(r)
    
    # print(x_cholesky)
    # print(x_qr)
    
    A = question_4(5)
    print(cholesky.check_symmetric(A))
    print(cholesky.check_spd(A))