function d = gauss_elm(k, p)
% Gaussian elimination for solving linear system of equation
% i.e., [k]{d} = {p} where k is a square matrix and both d and p are
% column vectors. (1) The first loop k matrix is transformed to an
% upper triangle matrix and (2) in the next phase the system of
% equations is solved using backward substitution approach.

% inpout(s)
% k : square matrix of unknown coefficients
% p : column vector of linear system of equation solutions

% output(s)
% d : column vector of solutions

% combining k and p
k = [k p];

% number of rows
n = size(k, 1);

% transorming k to an upper triangle
for ii = 1:n - 1
    for jj = 1:(n - ii - 1) % iterate over rows
        p = -k(jj + ii, ii) / k(ii, ii); % pivot value
        k(jj + ii, :) = p * k(ii, :) + k(jj + ii, :);
    end
end

% the last coefficient solution (d[n] = p[n] / k[n, n])
d(n) = k(n, n + 1) / k(n, n);

% backward substition
for ii = (n - 1):-1:1 % moving substition in rows
    a = k(ii, n + 1); % creating an intermediate variable instead of p
    for jj = (ii + 1):n % p assembly
        a = a - k(ii, jj) * d(jj);
    end
    d(ii) = a / k(ii, ii); % solving for d[n - 1]
end