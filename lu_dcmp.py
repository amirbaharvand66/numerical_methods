import numpy as np


def lu_dcmp(k):
    """
    LU decomposition
    [k] = [L][U]
        - k matrix is decomposed into two lower (L) and upper (U) matrices
        - L matrix is indeed formed from an identity matrix
        - U is the upper triangular matrix from Gaussian elimination
    
    inpout(s)
    k : square matrix of unknown coefficients
    
    output(s)
    L : lower triangular matrix
    U : upper triangular matrix
    
    originally coded by Amir Baharvand(AB) (09-20)
    """
    # number of rows
    n = k.shape[0]
    
    # initializing U
    U = k;
    
    # lower triangular matrix (L)
    L = np.eye(n);
    
    # upper triangular matrix (U)
    for ii in range(n - 1):
        for jj in range(1, n - ii): # iterate over rows
            piv = U[jj + ii, ii] / U[ii, ii]; # pivot value
            U[jj + ii, :] = -piv * U[ii, :] + U[jj + ii, :];
            L[jj + ii, ii] = piv;
    
    return L, U


def lu_dcmp_sol(L, U, p):
    """
    LU decomposition solver
    solving linear system of equations using LU decomposition 
    (1) [k] = [L][U]
        L and U are the outputs from LU decomposition
    (2) [k]{d} = {p}
        [L][U]{d} = {p} where [U]{d} = {Y}
        in linear system of equation< k is substituted by the L and U matrices
    (3) solve for d in two steps:
        (3.1) [L]{Y} = {p} where Y is solved from forward substitution
        3.2) [U]{Y} = {Y} is solved by a backward substution
    
    % inpout(s)
    % L : lower triangular matrix
    % U : upper triangular matrix
    % p : column vector of linear system of equation solutions
    
    % output(s)
    % d : column vector of solutions
    
    % originally coded by Amir Baharvand(AB) (09-20)
    """
    # number of rows
    n = U.shape[0]
    
    # (2) creating an intermediate column vector, Y
    Y = np.zeros((n, 1))
    
    # (3.1) forward substitution
    Y[0] = p[0] / L[0, 0] # solving for the first solution
    
    for ii in range(1 ,n):
        a = 0 # an intermediate variavble to hold found values for the solved
        for jj in range(ii): # a assembly
            a = a - L[ii, jj] * Y[jj]
        Y[ii] = p[ii] + a; # solving for Y[ii]
    
    # (3.2) backward substition
    d = np.zeros((n, 1), dtype = 'd'); # initializng d
    d[n - 1] = Y[n - 1] / U[n - 1, n - 1]; # solving for the last solution (d[n] = p[n] / k[n, n])
    
    for ii in range((n - 2), -1, -1): # substition in rows
        a = 0
        for jj in range(ii + 1, n): # a assembly
            a = a - U[ii, jj] * d[jj]
        d[ii] = (Y[ii] + a) / U[ii, ii] # solving for d[ii]
    
    return d