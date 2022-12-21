function [error] = xval_error(X, Y, lambda, part)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(X, Y, SIGMA, PART)
%
% Returns the average N-fold cross validation error of the kernel regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used.
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KERNEL_REGRESSION

% FILL IN YOUR CODE HERE

n_folds = max(part);

e = 0;
e_i = 0;

for i = 1:n_folds
        Xtest = X([find(part == i)],:); %chosing the i'th set to test
        Ytest = Y([find(part == i)]);
        Xtrain = X([find(part ~= i)],:); 
        Ytrain = Y([find(part ~= i)]); 
        
        ypred1 = predict_labels(Xtrain,Ytrain,Xtest,lambda); 
        
  
        e_i  = e_i + immse(ypred1, Ytest);  
end

error = mean(e_i); % cross validation error

end