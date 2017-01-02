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
sigma = 0.1;

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

loop = 0;
for C = [0.01 0.03 0.1 0.3 1 3 10 30]
  for sigma = [0.01 0.03 0.1 0.3 1 3 10 30]
    loop = loop + 1;
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    predict_err(loop) = mean(double(predictions ~= yval));
    C_ind(loop) = C;
    sigma_ind(loop) = sigma;
    fprintf(['predicted error = %f with C = %f and sigma = %f\n'], ...
	predict_err(loop), C, sigma);
  end
end

[min_val min_loc] = min(predict_err);

fprintf(['Smallest predicted error = %f with C = %f and sigma = %f\n'], min_val, ...
      C_ind(min_loc), sigma_ind(min_loc));

% output
% Smallest predicted error = 0.030000 with C = 1.000000 and sigma = 0.100000

C = C_ind(min_loc);
sigma = sigma_ind(min_loc);

% =========================================================================

end
