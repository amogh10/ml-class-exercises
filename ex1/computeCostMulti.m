function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2); % number of features

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

total_sum = 0;
for i=1:m,
  % feature data
  x           = X(i, :)';
  % h(0) = 0tx
  hypothesis  = theta' * x;
  cost_sum    = (hypothesis - y(i))^2;
  total_sum   = total_sum + cost_sum;
end

J = total_sum / (2*m);

% =========================================================================

end
