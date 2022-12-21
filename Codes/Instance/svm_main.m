clear all;
close all;
%Import Data
load training_data.mat; %outputs train_inputs and train_labels
part = make_xval_partition(size(train_inputs, 1), 5)';
e = 0;
e_i = 0;
n_folds = max(part);
for i = 1:n_folds
        Xtest = train_inputs([find(part == i)],:); %chosing the i'th set to test
        Ytest = train_labels([find(part == i)],:);
        Xtrain = train_inputs([find(part ~= i)],:); 
        Ytrain = train_labels([find(part ~= i)],:); 
        ypred1 = svm_predict(Xtrain,Ytrain,Xtest);
        e_i(i)  = error_metric(ypred1, Ytest);
end
cv_error = mean(e_i) % cross validation error



te_inputs = [train_inputs(:, 1:21) log(train_inputs(:, 22:end)) ];
bc_svm_full = [726 138.6 4.78 65.5 18.5 20.8 31.0 75.7 15.9];
ks_svm_full = [96.3 126.3 105.3 107.0 64.5 133.6 81.6 90.5 95.7];
[Z_train,mu_train,sigma_train] = zscore(te_inputs,0,1);
for ii = 1:9
    model= fitrsvm(Z_train, train_labels(:,ii), 'KernelFunction', 'Gaussian','BoxConstraint', bc_svm_full(ii), 'KernelScale', ks_svm_full(ii));
    pred_labels(:,ii) = predict(model, Z_train);
end
train_error = error_metric(pred_labels, train_labels)