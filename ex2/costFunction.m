function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J     = 0;
grad  = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Calculate J(theta)

  cost_sum = 0;
  for i=1:m,
    x   = X(i, :)';
    hx  = sigmoid(theta' * x);

    cost_sum = cost_sum + (-1 * y(i) * log(hx)) - ((1 - y(i)) * log(1 - hx));
  end

  J = cost_sum / m;

% Calculate gradients for each theta-i

  k = size(theta, 1);
  for j=1:k,
    sum_tmp = 0;
    for i=1:m,
      x = X(i, :)';
      h = sigmoid(theta' * x);
      sum_tmp = sum_tmp + ((h - y(i)) * X(i,j));
    end

    grad(j) = sum_tmp / m;
  end

% =============================================================

end
