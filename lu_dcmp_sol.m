function [d] = lu_dcmp_sol(L, U, p)
% LU decomposition solver
% solving linear system of equations using LU decomposition 
% (1) [k] = [L][U]
%       L and U are the outputs from LU decomposition
% (2) [k]{d} = {p}
%       [L][U]{d} = {p} where [U]{d} = {Y}
%       in linear system of equation< k is substituted by the L and U matrices
% (3) solve for d in two steps:
%       (3.1) [L]{Y} = {p} where Y is solved from forward substitution
%       (3.2) [U]{Y} = {Y} is solved by a backward substution

% inpout(s)
% L : lower triangular matrix
% U : upper triangular matrix
% p : column vector of linear system of equation solutions

% output(s)
% d : column vector of solutions

% originally coded by Amir Baharvand(AB) (09-20)

% number of rows
n = size(U, 1);

% (2) creating an intermediate column vector, Y
Y = zeros(n, 1);

% (3.1) forward substitution
Y(1) = p(1) / L(1); % solving for the first solution
for ii = 2:n
    a = 0; % an intermediate variavble to hold found values for the solved
    for jj = 1:(ii - 1) % a assembly
        a = a - L(ii, jj) * Y(jj);
    end
    Y(ii) = p(ii) + a; % solving for Y[ii]
end

% (3.2) backward substition
d = zeros(n, 1); % initializng d
d(n) = Y(n) / U(n, n); % solving for the last solution (d[n] = p[n] / k[n, n])

for ii = (n - 1):-1:1 % substition in rows
    a = 0;
    for jj = (ii + 1):n % a assembly
        a = a - U(ii, jj) * d(jj);
    end
    d(ii) = (Y(ii) + a) / U(ii, ii); % solving for d[ii]
end