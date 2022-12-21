clear all;
close all;
%Import Data
load training_data.mat; %outputs train_inputs and train_labels

part = make_xval_partition(size(train_inputs, 1), 5)';
n_folds = max(part);
e = 0;
e_i = 0;
min_error = 100000;
min_k = 100000;
k = 1:5:101;
for ii = 1:size(k,2)
    for i = 1:n_folds
            Xtest = train_inputs([find(part == i)],:); %chosing the i'th set to test
            Ytest = train_labels([find(part == i)],:);
            Xtrain = train_inputs([find(part ~= i)],:); 
            Ytrain = train_labels([find(part ~= i)],:); 
            ypred1 = k_means_method(Xtrain,Ytrain,Xtest,k(ii));
            e_i(i)  = error_metric(ypred1, Ytest);
    end
    error(ii) = mean(e_i); % cross validation error
    if(error(ii) < min_error)
        min_k = k(ii);
        min_error = error(ii);
    end
end

ideal_k_value = min_k
cv_error = min_error
ypred1 = k_means_method(train_inputs,train_labels,train_inputs,min_k);
training_error  = error_metric(ypred1, train_labels)

plot(k, error);