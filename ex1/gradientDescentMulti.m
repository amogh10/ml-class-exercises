function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

  % ====================== YOUR CODE HERE ======================
  % Instructions: Perform a single gradient step on the parameter vector
  %               theta.
  %
  % Hint: While debugging, it can be useful to print out the values
  %       of the cost function (computeCostMulti) and gradient here.
  %

  % number of features
  n = size(X, 2);

  % copy vector theta
  theta_tmp = theta;

  % iterate over each theta
  for j=1:n,

    % compute: sum( h(x(i)) - y(i) ) for 1 <= i <= m
    sum_part = 0;
    for i=1:m,
      x = X(i, :)';
      h = theta' * x;

      sum_part = sum_part + ((h - y(i)) * X(i,j));
    end

    % compute theta(j)
    theta_tmp(j) = theta_tmp(j) - (alpha * (sum_part / m));
  end

  % copy over theta_tmp to final theta
  theta = theta_tmp;

  % ============================================================

  % Save the cost J in every iteration
  J_history(iter) = computeCostMulti(X, y, theta);

end

end
