import numpy as np
import math

# Generate Random nxn matrix
def generate_random_spd(n):
    A = np.random.rand(n,n)
    spd = np.dot(A,A.T)
    spd += np.eye(n) * 1e-6
    return spd
    
def check_symmetric(A):
    A = np.array(A)
    print(A.T)
    if np.array_equal(A,A.T):
        return True
    return False
        
# Check if a matrix is Symmetric and Positive Definite
def check_spd(A):
    if check_symmetric(A):
        # Check for Positive Definite xTAx > 0, unless x = 0, or all EigenValues are greater than 0
        eigenvalues = np.linalg.eigvals(A)
        if np.all(eigenvalues > 0):
            return True
        else:
            return False
    return False

# Cholesky Decomposition
def cholesky(A):
    try:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                summation = 0
                if i == j:
                    for k in range(j):
                        summation +=  A[j,k] ** 2
                    A[j,j] = math.sqrt(A[j,j] - summation)  
                elif i > j:
                    for k in range(j):
                        summation += A[i,k] * A[j,k]
                    A[i,j] = 1/A[j,j] * (A[i,j] - summation)
                else:
                    A[i,j] = 0
        return A
    except Exception as e:
        return print("This is not an SPD Matrix")
    
def cholesky_det(A):
    curr = 1
    L = cholesky(A)
    for i in range(L.size[0]):
        curr += curr * L[i,i]
    return curr ** 2


def cholesky_solve_forward(A,b):
    L = cholesky(A)
    # Ly = b
    y = np.array([])
    # Compute y 
    for i in range(len(b.size[0])):
        print(i)
        if i == 0:
            np.append(y,b[0])
        else:
            # This is worng
            np.append(y,b[i] - np.dot(L[i],y) / L[i,i])
    # Compute x 
    x = np.array([])
    return x


def cholesk_solve_backward(A,x,b):
    return

if __name__ == "__main__":
    print("START TEST")
    A = generate_random_spd(4)
    Original = A.copy()
    #A = np.matrix([[2,1],[1,2]],dtype=np.float64)
    #print(A)
    A = cholesky(A)
    decomp_check = np.matmul(A,A.T)
    print("L Decomp")
    print(A)
    print("CHOLESKY ")
    print(decomp_check)
    print("ORIGINAL")
    print(Original)
    print(np.allclose(decomp_check,Original))
    