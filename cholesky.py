import numpy as np
import math

# Generate Random nxn matrix
def generate_random_spd(n):
    A = np.random.randint(0,10,[n,n])
    A = A + A.T
    return A + n * np.eye(n)


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
        if min(eigenvalues) < 1:
            return False
        else:
            return True
    return False


# Central program
def cholesky_factor(A):
    print(A)
    row,col = A.shape
    for i in range(row):
        for j in range(col):
            if i == j:
                if i > 0:
                    # Retrieve every item =to the left of i,j
                    current = A[i,:j]
                    print(A)
                    print(A[i,j])
                    print(current)
                    current = current ** 2
                    print(current)
                    # Diagonal
                    A[i,j] = math.sqrt(A[i,j] - np.sum(current))
                else:
                    A[i,j] = math.sqrt(A[i,j])
            else:
                if j < i:
                    print("J < I")
                    #retrieve every item in the row above
                    row_i = A[i,:j]
                    row_j = A[j,:j]
                    print(i)
                    print(j)
                    print(row_i)
                    print(row_j)
                    if len(row_i) > 0:
                        summation = np.dot(row_i,row_j)
                    else:
                        summation = 0
                    print(summation)
                    A[i,j] = (A[i,j] - summation)/ A[j,j]
    return A

if __name__ == "__main__":
    print("START TEST")
    # A = generate_random_spd(4)
    # print(check_spd(A))
    # print(A)
    A = np.matrix([[2,1],[1,2]])
    print(A)
    cholesky_factor(A)
    print(A)