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
    if check_symmetric(A):
        try:
            cholesky(A)
            return True
        except:
            # Return False if Cholesky returns an exception
            return False
    # Return False if matrix is not Symmetric
    return False

# Cholesky Decomposition
def cholesky(A):
    try:
        # Iterate through rows i and columns j
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                summation = 0
                print(A)
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
        print("HERE")
        return A
    # Exception if matrix is not SPD
    except Exception as e:
        return print("This is not an SPD Matrix")

# Calculate determinant of Cholesky Matrix CHECK THIS ---------------------------------------------------
def cholesky_det(A):
    curr = 1
    L = cholesky(A)
    for i in range(L.size[0]):
        curr += curr * L[i,i]
    return curr ** 2

# Forward Substitution L * y = b
def cholesky_forward_sub(A,b):
    L = cholesky(A)
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
def cholesk_backward_sub(L,y):
    x = np.array([])
    # L.T * x = y
    # Compute x, same logic as forward substitution
    for i in range(len(y)):
        if i == 0:
            x = np.append(x,y[i] / L.T[i,i])
        else:
            computed = L.T[i,:i] @ x
            x_i = (y[i] - computed) / L.T[i,i]
            x = np.append(x,x_i)
    # Reshape x to column array
    return x.reshape(-1,1)

if __name__ == "__main__":
    # print("START TEST")
    # A = generate_random_spd(4)
    # Original = A.copy()
    #A = np.matrix([[2,1],[1,2]],dtype=np.float64)
    #print(A)
    # A = cholesky(A)
    # decomp_check = np.matmul(A,A.T)
    # print("L Decomp")
    # print(A)
    # print("CHOLESKY ")
    # print(decomp_check)
    # print("ORIGINAL")
    # print(Original)
    # print(np.allclose(decomp_check,Original))
    # Testing Forward Test----------------------------
    print("Testing Forward")
    A = generate_random_spd(3)
    Original_A = A.copy()
    b = np.array([1,1,1])
    b = b.reshape(-1,1)
    y = cholesky_forward_sub(A,b)
    x = cholesk_backward_sub(A,y)
    print(x)
    print(np.matmul(Original_A,x))