function [L, U] = lu_dcmp(k)
% LU decomposition
% [k] = [L][U]
%       - k matrix is decomposed into two lower (L) and upper (U) matrices
%       - L matrix is indeed formed from an identity matrix
%       - U is the upper triangular matrix from Gaussian elimination

% inpout(s)
% k : square matrix of unknown coefficients

% output(s)
% L : lower triangular matrix
% U : upper triangular matrix

% originally coded by Amir Baharvand(AB) (09-20)

% number of rows
n = size(k, 1);

% initializing U
U = k;

% (1.1) lower triangular matrix (L)
L = eye(n);

% (1.2) upper triangular matrix (U)
for ii = 1:n - 1
    for jj = 1:(n - ii) % iterate over rows
        piv = U(jj + ii, ii) / U(ii, ii); % pivot value
        U(jj + ii, :) = -piv * U(ii, :) + U(jj + ii, :);
        L(jj + ii, ii) = piv;
    end
end