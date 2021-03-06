function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute regularized cost function J
% =========================================================================

% h(x)
hx = X * theta;

% Cost term
cost = sum((hx - y) .^ 2) / (2*m);

% Regularization term
reg = ((lambda/(2*m)) * sum(theta(2:end) .^ 2));

% J
J = cost + reg;

% Compute gradients
% =========================================================================

% Derivatives for each theta
deriv = (X' * (hx - y)) / m;

% Regularization terms
reg_derivs = (lambda/m) .* theta;
reg_derivs(1) = 0; % don't regulariza theta for first index (theta-0)

% grad
grad = deriv + reg_derivs;

% =========================================================================

grad = grad(:);

end
