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

# Calculate determinant of Cholesky Matrix, Assumes A is already in Choleksy Decomposition
def cholesky_det(A):
    curr = 1
    for i in range(len(A)):
        curr *= A[i,i]
    return curr ** 2

# Forward Substitution L * y = b
# Cannot Compute L.T
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
            computed = np.dot(L[i+1:,i] , x[i+1:])
            x[i] = (y[i,0] - computed) / L[i,i]
    # Reshape x to column array
    return x.reshape(-1,1)

# Assumes A is not in cholesky Format
def cholesky_inv(A):
    cholesky(A)
    # A^-1 = [a1, a2, a3, a4, a5]
    inv_A = np.empty((len(A),0))
    for i in range(len(A)):
        e_i = np.zeros(len(A))
        e_i[i] = 1
        e_i = e_i.reshape(-1,1)
        # L * y = e
        y_i = cholesky_forward_sub(A,e_i)
        # L.T * a = y
        a_i = cholesky_backward_sub(A,y_i)
        inv_A = np.concatenate((inv_A, a_i), axis=1)
    return inv_A

if __name__ == "__main__":
    print("Test Functions -----------------------------------------------------------------------")
    print("Generate a random SPD Matrix")
    A = generate_random_spd(5)
    # Since A gets overwritten we need to save the original to check the inverse calculation
    A_original = A.copy()
    print(A)
    print("Check if its SPD")
    print("Matrix is SPD:" , check_spd(A))
    print("Generate Cholesky")
    cholesky(A)
    print(A)
    print("Calculate Determinant:", cholesky_det(A))
    print("Test Forward and Backwards Sub")
    x = np.ones(5)
    x.reshape(-1,1)
    print("x:", x)
    y = cholesky_forward_sub(A,x)
    print("y: ", y)
    b = cholesky_backward_sub(A,y)
    print("b: ", b)
    print("Compute Inverse of A")
    # This does not take extra space but it needs to be saved elsewhere for testing
    print(A)
    # Calculate the inverse of a copy of the original matrix
    inv_A = cholesky_inv(A_original.copy())
    print(inv_A)
    # Check if computer properly 
    print("A * A^-1")
    print(np.round(A_original @ inv_A))