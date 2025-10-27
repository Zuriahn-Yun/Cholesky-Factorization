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
    cholesky.cholesky(A)
    L = A
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
    cholesky.cholesky(BT_B)
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
        p = 6
        n = 6
        
        B = create_spd_pascal(p,n)
        print("N X N, SPD Pascal Matrix, p = ", p,", n =", n)
        print(B)
        det_B = determinant_spd_pascal(B)
        print("Determinant of B = ", det_B)
    
    print("#2 ------------------------------------------------------------------------------------------------------------------")
    B = create_spd_pascal(p,n)
    print(B)
    B = 1/2 * (B @ B.T)
    B_original = B.copy()
    print("B")
    print(B)
    B_inv = cholesky.cholesky_inv(B.copy())
    print("Is inv(B) SPD: ", cholesky.check_spd((B_inv.copy())))
    print("This returns False but should return True")
    print("If we round B and reverse its Operations, the Check will Return True")
    print("Operations: ")
    print("B_inv = np.round(B_inv,12)")
    print("B_inv = (B_inv + B_inv.T) / 2")
    B_inv = np.round(B_inv,12)
    B_inv = (B_inv + B_inv.T) / 2
    print("Is inv(B) SPD: ", cholesky.check_spd((B_inv)))    

    print("#3 ------------------------------------------------------------------------------------------------------------------")
    print("Small r Value")
    r = 3
    x = least_squares(r)
    print("r = ", r)
    print("Least Squares x")
    print(x)
    qr_x = QR_least_squares(r)
    print("QR Least Squares x")
    print(qr_x)
    print("DIfference")
    print(qr_x - x)
    print("Larger r Values")
    for i in range(8,14):
        print("Iteration #", i - 7)
        r = 10 ** i
        x = least_squares(r)
        print("r = ", r)
        print("Least Squares x")
        print(x)
        qr_x = QR_least_squares(r)
        print("QR Least Squares x")
        print(qr_x)
        print("DIfference")
        print(qr_x - x)
    print("#4 ------------------------------------------------------------------------------------------------------------------")
    print("IF the matrix is even it is 100 x 100, if its odd its 99 x 99")
    print("Even, All Positive: " , cholesky.check_spd(question_4(100,1,1)))
    print("Even, Negative Corners: ", cholesky.check_spd(question_4(100,-1,1)))
    print("Even, Negative Diagonals: ", cholesky.check_spd(question_4(100,1,-1)))
    print("Even, All Negative: " ,cholesky.check_spd(question_4(100,-1,-1)))
    print("Odd, All Positive: ", cholesky.check_spd(question_4(99,1,1)))
    print("Odd, Negative Corners: " ,cholesky.check_spd(question_4(99,-1,1)))
    print("Odd, Negative Diagonals: " ,cholesky.check_spd(question_4(99,1,-1)))
    print("Odd, All  Negative: " ,cholesky.check_spd(question_4(99,-1,-1)))
    
    print("Matrix of size 300 x 300 if even and 299 x 299 if odd.")
    print("Even, All Positive: " , cholesky.check_spd(question_4(300,1,1)))
    print("Even, Negative Corners: ", cholesky.check_spd(question_4(300,-1,1)))
    print("Even, Negative Diagonals: ", cholesky.check_spd(question_4(300,1,-1)))
    print("Even, All Negative: " ,cholesky.check_spd(question_4(300,-1,-1)))
    print("Odd, All Positive: ", cholesky.check_spd(question_4(299,1,1)))
    print("Odd, Negative Corners: " ,cholesky.check_spd(question_4(299,-1,1)))
    print("Odd, Negative Diagonals: " ,cholesky.check_spd(question_4(299,1,-1)))
    print("Odd, All  Negative: " ,cholesky.check_spd(question_4(299,-1,-1)))