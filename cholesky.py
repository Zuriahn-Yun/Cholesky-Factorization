import numpy as np
import math

# Generate Random nxn matrix
def generate_random(n):
    return np.random.rand(n,n)


def check_symmetric(A):
    A = np.array(A)
    if A == A.T:
        return True
    return False
        

# Check if a matrix is Symmetric and Positive Definite
def check_spd(A):
    if check_symmetric(A):
        # Check for Positive Definite xTAx > 0, unless x = 0, or all EigenValues are greater than 0
        eigenvalues = np.linalg.eigvals(A)
        if min(eigenvalues) < 1:
            return False
        else:
            return True
    return False

# Central program
def cholesky_factor(A):
    row,col = A.shape
    for i in range(row):
        for j in range(col):
            if i == j:
                # Retrieve every item directly above index i,j
                col = A[:i,j] ** 2
                # Diagonal
                A[i,j] = math.sqrt(A[i,j] - col)
    return

if __name__ == "__main__":
    print("START TEST")
    A = generate_random(4)
    cholesky_factor(A)
    print(A)