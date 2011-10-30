function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% n
n = size(theta, 1);

% Calculate J(theta)

  cost_sum = 0;
  for i=1:m,
    x   = X(i, :)';
    hx  = sigmoid(theta' * x);

    cost_sum = cost_sum + (-1 * y(i) * log(hx)) - ((1 - y(i)) * log(1 - hx));
  end

  theta_sum = 0;
  for j=2:n,
    theta_sum = theta_sum + (theta(j) * theta(j));
  end
  reg = (lambda/(2*m)) * theta_sum;

  J = (cost_sum / m) + reg;

% Calculate gradients for each theta-i

  for j=1:n,
    error_sum = 0;
    for i=1:m,
      x = X(i, :)';
      h = sigmoid(theta' * x);

      error_sum = error_sum + ((h - y(i)) * X(i,j));
    end

    if j >= 2
      error_sum = error_sum + (lambda * theta(j));
    end

    grad(j) = error_sum / m;
  end

% =============================================================

end
