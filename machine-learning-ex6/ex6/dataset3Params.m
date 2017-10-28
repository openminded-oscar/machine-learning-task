function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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
%C_SIGMA_RANGE = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%maxI = 0;
%maxJ = 0;
%max = 0;
%for i = 1:size(C_SIGMA_RANGE, 2)
%  C=C_SIGMA_RANGE(i);
%  for j=1:size(C_SIGMA_RANGE, 2)
%    sigma = C_SIGMA_RANGE(j);
%    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%    validatingResult = (svmPredict(model, Xval)==yval);
%    count1 = 0;
%    for z=1:size(validatingResult,1)
%      if (validatingResult(z,1)==1)
%        ++count1;
%      endif
%    endfor
%    if count1>max
%      max = count1;
%      maxI = i;
%      maxJ = j;
%    endif
%  endfor
%endfor
%C = C_SIGMA_RANGE(maxI);
%sigma = C_SIGMA_RANGE(maxJ);

C = 1;
sigma = 0.1;
% =========================================================================

end
