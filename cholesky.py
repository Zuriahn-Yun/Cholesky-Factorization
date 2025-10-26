import numpy as np
import math

# Generate Random nxn matrix
def generate_random_spd(n):
    # Create Random n x n Matrix 
    A = np.random.rand(n,n)
    spd = np.dot(A,A.T)
    spd += np.eye(n) * 1e-6
    return spd

# Generate Random Matrix for Testing
def generate_random(n):
    return np.random.rand(n,n)

# Check if Matrix is Symmetric
def check_symmetric(A):
    A = np.array(A)
    if np.array_equal(A,A.T):
        return True
    return False
        
# Check if a matrix is Symmetric and Positive Definite using Cholesky
def check_spd(A):
    # If symmetric try Cholesky
    if not check_symmetric(A):
        return False
    status = cholesky(A)
    if status == "Matrix is not SPD":
        return False
    return True

# Cholesky Decomposition
def cholesky(A):
    try:
        # Iterate through rows i and columns j
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                summation = 0
                # Compute Diagonals
                if i == j:
                    for k in range(j):
                        summation +=  A[j,k] ** 2
                    A[j,j] = math.sqrt(A[j,j] - summation)  
                # Compute Lower Triangle Values
                elif i > j:
                    for k in range(j):
                        summation += A[i,k] * A[j,k]
                    A[i,j] = 1/A[j,j] * (A[i,j] - summation)
                # Otherwise the values are 0
                else:
                    A[i,j] = 0
        return "Matrix is SPD"
    # Exception if matrix is not SPD
    except Exception as e:
        return "Matrix is not SPD"

# Calculate determinant of Cholesky Matrix CHECK THIS ---------------------------------------------------
def cholesky_det(A):
    curr = 1
    cholesky(A)
    L = A
    for i in range(L.size[0]):
        curr *= L[i,i]
    return curr ** 2

# Forward Substitution L * y = b
def cholesky_forward_sub(L,b):
    # Ly = b
    y = np.array([])
    # Compute y 
    for i in range(len(b)):
        if i == 0:
            y = np.append(y,b[i] / L[i,i])
        else:
            # Since we are appending to y as we go all of y is already computed
            # We can dot y with every L value in this row up to but not including i 
            computed = L[i,:i] @ y
            y_i = (b[i] - computed) / L[i,i]
            y = np.append(y,y_i)
    # Reshape y to a Column Array
    return y.reshape(-1,1)

# Backwards Substitution, L.T * x = y
# y is a Column Vector 
def cholesky_backward_sub(L,y):
    x = np.zeros(len(y))
    # L.T * x = y
    # Compute x, same logic as forward substitution but backwards
    for i in range(len(y) -1 , -1 ,-1):
            computed = L.T[i,i+1:] @ x[i+1:]
            x[i] = (y[i,0] - computed) / L.T[i,i]
    # Reshape x to column array
    return x.reshape(-1,1)

def cholesky_inv(A):
    cholesky(A)
    L = A
    # A^-1 = [a1, a2, a3, a4, a5]
    inv_A = np.empty((len(A),0))
    for i in range(len(A)):
        e_i = np.zeros(len(A))
        e_i[i] = 1
        e_i = e_i.reshape(-1,1)
        # L * y = e
        y_i = cholesky_forward_sub(L,e_i)
        # L.T * a = y
        a_i = cholesky_backward_sub(L,y_i)
        inv_A = np.concatenate((inv_A, a_i), axis=1)

    return inv_A

if __name__ == "__main__":
    print("Test Functions -----------------------------------------------------------------------")
    B = np.array([[16,4,8,4],
                  [4,10,8,4],
                  [8,8,12,10],
                  [4,4,10,12]])
    cho_inv_b = cholesky_inv(B.copy())
    print(np.matmul(B,cho_inv_b))
    
    