function [pred_labels] = svm_predict(train_inputs,train_labels,test_inputs)

train_inputs(:,22:end) = log (train_inputs(:,22:end));
test_inputs(:,22:end) = log (test_inputs(:,22:end));
[Z_train,mu_train,sigma_train] = zscore(train_inputs,0,1);
Z_test = (test_inputs-mu_train)./sigma_train;

bc_svm_full = [726 138.6 4.78 65.5 18.5 20.8 31.0 75.7 15.9];
ks_svm_full = [96.3 126.3 105.3 107.0 64.5 133.6 81.6 90.5 95.7];

for ii = 1:9
    model= fitrsvm(Z_train, train_labels(:,ii), 'KernelFunction', 'Gaussian','BoxConstraint', bc_svm_full(ii), 'KernelScale', ks_svm_full(ii));
    pred_labels(:,ii) = predict(model, Z_test);
end

end