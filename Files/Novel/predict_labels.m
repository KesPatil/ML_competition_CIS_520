function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)
train_inputs(:,22:end) = log (train_inputs(:,22:end));
test_inputs(:,22:end) = log (test_inputs(:,22:end));
 
 
[Z_train,mu_train,sigma_train] = zscore(train_inputs,0,1);
Z_test = (test_inputs-mu_train)./sigma_train;
 
[Z_freqs mu_freqs sigma_freqs] = zscore(train_inputs(:, 22:end));
[Z_freqs2 mu_freqs2 sigma_freqs2] = zscore(test_inputs(:, 22:end));
 
idx = kmeans(Z_freqs', 30)';
kmzf = zeros(size(Z_freqs, 1), max(max(idx)));
for ii = 1:max(max(idx))
    mat = Z_freqs(:,idx == ii);
    kmzf(:, ii) = mean(mat,2);
end
kmzf = [Z_train(:,1:21), kmzf];
 
kmzf2 = zeros(size(Z_freqs2, 1), max(max(idx)));
for ii = 1:max(max(idx))
    mat2 = Z_freqs2(:,idx == ii);
    kmzf2(:, ii) = mean(mat2,2);
end
kmzf2 = [Z_test(:,1:21), kmzf2];
 
bc_svm_full = [726 138.6 4.78 65.5 18.5 20.8 31.0 75.7 15.9];
ks_svm_full = [96.3 126.3 105.3 107.0 64.5 133.6 81.6 90.5 95.7];
 
bc_svm_reg = [974 945 2.90 72.87 7.22 9.28 666 25.5 225.8];
ks_svm_reg = [14.0 32.0 8.45 14.18 5.66 6.87 44.1 18.2 28.9]; 
 
 
lambda = [0.00137 0.01595 0.11042 0.18517 0.05411 0.01615 0.01990 0.05756 0.03365];
lambda2 = [0.02 0.1 2 0.3 0.5 0.5 0.5 1.8 1.0];
 
k = [12 15 17 16 18 14 18 17 15];
 
 
catidx = 7:10; % also column 21
catidx = [catidx 21];
 
 
for ii = 1:9
     
    %%Gaussian Kernel SVM on Data
    svm_full = fitrsvm(Z_train, train_labels(:,ii), 'KernelFunction', 'Gaussian','BoxConstraint', bc_svm_full(ii), 'KernelScale', ks_svm_full(ii));
    svm_reg = fitrsvm(Z_train(:, 1:21), train_labels(:,ii), 'KernelFunction', 'Gaussian','BoxConstraint', bc_svm_reg(ii), 'KernelScale', ks_svm_reg(ii));
    svm_km = fitrsvm(kmzf, train_labels(:,ii), 'KernelFunction', 'Gaussian','BoxConstraint', bc_svm_reg(ii), 'KernelScale', ks_svm_reg(ii));   
    svm_predictions = mean([predict(svm_full,Z_test) predict(svm_reg,Z_test(:, 1:21)) predict(svm_km,kmzf2)],2);
 
    %%PLS Fit
    [~,~,~,~,betaPLSk,~] = plsregress([train_inputs(:, 1:21) train_inputs(:, 22:end)],train_labels(:,ii),k(ii));
    [n,p] = size(Z_test);
    pls_predictions = [ones(n,1) [test_inputs(:, 1:21) test_inputs(:, 22:end)]]*betaPLSk;
     
     
%     lin_full = fitrlinear(Z_train, train_labels(:,ii), 'Lambda', lambda2(ii),'Regularization','ridge');    
%     lin_reg = fitrlinear(Z_train(:, 1:21), train_labels(:,ii));
%     lin_km = fitrlinear(kmzf, train_labels(:,ii));
%     lin_predictions = mean([predict(lin_full,Z_test) predict(lin_reg,Z_test(:, 1:21)) predict(lin_km,kmzf2)],2);   
  
%     tree_full = fitrtree(Z_train, train_labels(:,ii));
%     tree_reg = fitrtree(Z_train(:, 1:21), train_labels(:,ii));
%     tree_km = fitrtree(kmzf, train_labels(:,ii));
%     tree_predictions = mean([predict(tree_full,Z_test) predict(tree_reg,Z_test(:, 1:21)) predict(tree_km,kmzf2)],2);
 
%     [B,FitInfo] = lasso(kmzf,train_labels(:, ii),'Alpha',0.5,'CV',2,'Options',statset('UseParallel',true));
%     idxLambda1SE = FitInfo.Index1SE;
%     coef = B(:,idxLambda1SE);
%     coef0 = FitInfo.Intercept(idxLambda1SE);
%     lasso_predictions = kmzf2*coef + coef0;
     
    ensemble_full = fitrensemble(train_inputs,train_labels(:,ii),'Method','LSBoost','NumLearningCycles',250,'LearnRate',0.1,'ResponseName','Symboling','CategoricalPredictors',catidx);
    ensemble_predictions = predict(ensemble_full,test_inputs);
     
    pred_labels(:,ii) = mean([ensemble_predictions svm_predictions pls_predictions],2); 
end
 
 
end