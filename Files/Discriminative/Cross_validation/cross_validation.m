clc;
clear all;

X = load('training_data.mat');
X_train = X.train_inputs;
Y_train = X.train_labels;


X_z = X_train;
 


Y_z = Y_train;

% Breaking into training (800) and testing data (219)

X_z_train = X_z(1:800,:);
X_z_test = X_z(801:1019,:);

Y_z_train = Y_z(1:800,:);
Y_z_test = Y_z(801:1019,:);


n  = size(X_z_train,1);
n_folds = 5;
part = make_xval_partition(n, n_folds);

% this lambda are changed manually to search for optimal lambda for
% different Y response columns
lambda = [0 1e-3 0.001 0.002 0.003 0.005 1e-2 0.02 0.05 ];


% Different lambda for different Y response columns 
%  choose the ones that gives lowest cross validation error
for i = 1:length(lambda)

error(i) = xval_error(X_z_train, Y_z_train(:,2),lambda(i), part);
end


scatter(lambda,error);



