
clc;
clear all;

X = load('training_data.mat');
X_train = X.train_inputs;
Y_train = X.train_labels;

X_z = X_train;
 


Y_z = Y_train;

% Breaking into training (800) and testing data (219)

X_z_train = X_z(1:700,1:end);
X_z_test = X_z(701:1019,1:end);

Y_z_train = Y_z(1:700,1:end);
Y_z_test = Y_z(701:1019,1:end);




train_inputs = X_z_train;
test_inputs = X_z_test;

train_labels = Y_z_train;

pred_labels=randn(size(test_inputs,1),size(train_labels,2));


% logarithmic transformation of the LDA data
train_inputs(:,22:end) = log (train_inputs(:,22:end));
test_inputs(:,22:end) = log (test_inputs(:,22:end));


% Standardization of the data
train_inputs = zscore(train_inputs);
test_inputs = zscore(test_inputs);


X_train_reconstruct = train_inputs;
X_test_reconstruct = test_inputs;



% the j's are the regularization parameters obtained through cross
% validation

j = 0.020;
mdl1 = fitrlinear(X_train_reconstruct,train_labels(:,1),'Regularization','ridge','Lambda',j);
ypred1 = predict(mdl1,X_test_reconstruct);
ypred1t = predict(mdl1,X_train_reconstruct);



j = 0.1;
mdl2 = fitrlinear(X_train_reconstruct,train_labels(:,2),'Regularization','ridge','Lambda',j);
ypred2 = predict(mdl2,X_test_reconstruct);
ypred2t = predict(mdl2,X_train_reconstruct);



j = 2;
mdl3 = fitrlinear(X_train_reconstruct,train_labels(:,3),'Regularization','ridge','Lambda',j);
ypred3 = predict(mdl3,X_test_reconstruct);
ypred3t = predict(mdl3,X_train_reconstruct);


j = 0.3;
mdl4 = fitrlinear(X_train_reconstruct,train_labels(:,4),'Regularization','ridge','Lambda',j);
ypred4 = predict(mdl4,X_test_reconstruct);
ypred4t = predict(mdl4,X_train_reconstruct);


j = 0.5;
mdl5 = fitrlinear(X_train_reconstruct,train_labels(:,5),'Regularization','ridge','Lambda',j);
ypred5 = predict(mdl5,X_test_reconstruct);
ypred5t = predict(mdl5,X_train_reconstruct);


j = 0.5;
mdl6 = fitrlinear(X_train_reconstruct,train_labels(:,6),'Regularization','ridge','Lambda',j);
ypred6 = predict(mdl6,X_test_reconstruct);
ypred6t = predict(mdl6,X_train_reconstruct);


j = 0.5;
mdl7 = fitrlinear(X_train_reconstruct,train_labels(:,7),'Regularization','ridge','Lambda',j);
ypred7 = predict(mdl7,X_test_reconstruct);
ypred7t = predict(mdl7,X_train_reconstruct);


j = 1.8;
mdl8 = fitrlinear(X_train_reconstruct,train_labels(:,8),'Regularization','ridge','Lambda',j);
ypred8 = predict(mdl8,X_test_reconstruct);
ypred8t = predict(mdl8,X_train_reconstruct);


j = 1.0;
mdl9 = fitrlinear(X_train_reconstruct,train_labels(:,9),'Regularization','ridge','Lambda',j);
ypred9 = predict(mdl9,X_test_reconstruct);
ypred9t = predict(mdl9,X_train_reconstruct);

pred_labels = [ypred1 ypred2 ypred3 ypred4 ypred5 ypred6 ypred7 ypred8 ypred9];
pred_labelst = [ypred1t ypred2t ypred3t ypred4t ypred5t ypred6t ypred7t ypred8t ypred9t];

ypred = pred_labels;
ypredt = pred_labelst;

error_test =  error_metric(ypred,Y_z_test)
error_train = error_metric(ypredt, Y_z_train)



