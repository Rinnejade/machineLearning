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
current_optim=100;
cur_c = 0.01;
while cur_c<30
	cur_sigma = 0.01;
	while cur_sigma<30
		cur_sigma*=3;
		model= svmTrain(X, y, cur_c, @(x1, x2) gaussianKernel(x1, x2, cur_sigma));
		predictions = svmPredict(model,Xval);
		test = mean(double(predictions~=yval));
		if(test<current_optim)
			current_optim = test;
			c = cur_c
			sigma = cur_sigma
		endif
	endwhile
	cur_c*=3;
endwhile






% =========================================================================

end
