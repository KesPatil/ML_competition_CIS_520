clc;
clear all;

X = load('training_data.mat');
X_train = X.train_inputs;
Y_train = X.train_labels;

X_z = X_train;
Y_z = Y_train;

% Breaking into training (800) and testing data (219)

X_z_train = X_z(1:800,1:end);
X_z_test = X_z(801:1019,1:end);

Y_z_train = Y_z(1:800,1:end);
Y_z_test = Y_z(801:1019,1:end);


train_inputs = X_z_train;
test_inputs = X_z_test;

train_labels = Y_z_train;






train_inputs(:,22:end) = log (train_inputs(:,22:end));
test_inputs(:,22:end) =   log (test_inputs(:,22:end));


X = train_inputs;
X_test = test_inputs;

% This k represents the number of components considered for partial least
% square regression. The components are the ones that preserve the maximum
% variation in response Y. The k was obtained by cross validation and
% found to be similar for all the y responses.
k = [1:20];



%for ii = 1:1
for i = 1:length(k)
    
y = Y_z_train(:,9);
[n,p] = size(X);
[~,~,~,~,betaPLSk,PLSPctVar] = plsregress(...
	X,y,k(i));
yfitPLS1(:,1) = [ones(n,1) X]*betaPLSk;

[n,p] = size(X_test);
y_test_fitPLS2(:,1)  = [ones(n,1) X_test]*betaPLSk;
e(i)=immse(y_test_fitPLS2(:,1), Y_z_test(:,9));
end
%end
[val, idx] = min(e);
disp(idx)
%scatter(k,e);



