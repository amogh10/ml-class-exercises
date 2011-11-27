function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% values to check for
candidates        = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% combine check values
[CVal, sigmaVal]  = meshgrid(candidates, candidates);
combinations      = [CVal(:) sigmaVal(:)];
m                 = size(combinations, 1);
% add a 3rd column to store the prediction error
error_measures    = [combinations zeros(m, 1)];

for i=1:m,
  % for each C, sigma combination train the SVM, get the prediction and compute
  % the error. Store it.
  C_curr    = error_measures(i, 1);
  sig_curr  = error_measures(i, 2);

  fprintf('\n[%d/%d] - Training for C=%.2f/sig=%.2f combination:', i, m, C_curr, sig_curr);

  model       = svmTrain(X, y, C_curr, @(x1, x2) gaussianKernel(x1, x2, sig_curr));
  predictions = svmPredict(model, Xval);
  pred_err    = mean(double(predictions ~= yval));

  error_measures(i, 3) = pred_err;
end

% get the minimum error and position
[min_err, min_err_idx] = min(error_measures(:,3), [], 1);

% assign computed C & sigma values
C     = error_measures(min_err_idx, 1);
sigma = error_measures(min_err_idx, 2);
fprintf('\n Best possible C/sigma combination found: C=%.2f, sigma=%.2f (w/err %.4f)', C, sigma, min_err);

% =========================================================================

end
