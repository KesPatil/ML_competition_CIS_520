function pred_labels=predict_labels(train_inputs,train_labels,test_inputs,j)
pred_labels=randn(size(test_inputs,1),size(train_labels,2));

%Take the mean of the traing data
train_inputs(:,1:21) = (train_inputs(:,1:21));
train_inputs(:,22:end) = (log(train_inputs(:,22:end)));

% train_inputs = train_inputs(:,2:end);
train_inputs = zscore(train_inputs);
%m = mean(train_inputs);

%Standardixe the data


%Take the mean centered data
%train_inputs  = train_inputs-m;


%Perform PCA
% [v,z,~] = pca(train_inputs);
%
% %New training data
% train_inputs = z*v';
% new_train_data = new_train_data(:,1:i);

%test_inputs = test_inputs(:,2:end);
test_inputs(:,1:21) = (test_inputs(:,1:21));
test_inputs(:,22:end) = (log(test_inputs(:,22:end)));
test_inputs = zscore(test_inputs);
 %m = mean(test_inputs);

%Take the mean centered data
%test_inputs  = test_inputs-m;


%Perform PCA
% [v,z,~] = pca(test_inputs);
%
% %New training data
% test_inputs = z*v';
% new_test_data = new_test_data(:,1:i);

%Loop through 9 predictions i.e create 9 models and predict 9 variables
%for i = 1:size(train_labels,2)
%     Lambda = logspace(-5,-1,15);
    %     model = fitrlinear(X_train_mean,train_labels(:,i));
    %     CVMdl =  fitrlinear(X_train_mean',train_labels(:,i),'ObservationsIn','columns','KFold',5,'Lambda',Lambda,...
    %         'Learner','leastsquares','Solver','sgd','Regularization','ridge');
    % %     numCLModels = numel(CVMdl.Trained);
    %     mse = kfoldLoss(CVMdl);
    %     [~,idxFinal] = min(log(mse));
    %     Mdl = fitrlinear(X_train_mean',train_labels(:,i),'ObservationsIn','columns','Lambda',Lambda,...
    %         'Learner','leastsquares','Solver','sgd','Regularization','ridge');
    %     Final_model = selectModels(Mdl,j);
    
    %train_labels(:,1) = train_labels(:,1) /100;
    %test_inputs(:,1) = test_inputs(:,1) / 100;
     Final_model = fitrlinear(train_inputs,train_labels, 'Regularization', 'ridge', 'Lambda', j);
% Final_model = lasso(train_inputs,train_labels(:,1),'Alpha',0.5);
    pred_labels = (predict(Final_model,test_inputs));
    %pred_labels = 100 * pred_labels;
    %     pred_labels(:,i) = (max(pred_labels(:,i)) - pred_labels(:,i))/(max(pred_labels(:,i))-min(pred_labels(:,i)));
%end





% %mdl1 = fitlm(X_z_train,Y_z_train(:,1));
% mdl1 = fitrlinear(train_inputs,train_labels(:,1));
% ypred1 = predict(mdl1,test_inputs);
%
%
% %mdl2 = fitlm(X_z_train,Y_z_train(:,2));
% mdl2 = fitrlinear(train_inputs,train_labels(:,2));
% ypred2 = predict(mdl2,test_inputs);
%
%
% %mdl3 = fitlm(X_z_train,Y_z_train(:,3));
% mdl3 = fitrlinear(train_inputs,train_labels(:,3));
% ypred3 = predict(mdl3,test_inputs);
%
%
% %mdl4 = fitlm(X_z_train,Y_z_train(:,4));
% mdl4 = fitrlinear(train_inputs,train_labels(:,4));
% ypred4 = predict(mdl4,test_inputs);
%
%
% %mdl5 = fitlm(X_z_train,Y_z_train(:,5));
% mdl5 = fitrlinear(train_inputs,train_labels(:,5));
% ypred5 = predict(mdl5,test_inputs);
%
%
% %mdl6 = fitlm(X_z_train,Y_z_train(:,6));
% mdl6 = fitrlinear(train_inputs,train_labels(:,6));
% ypred6 = predict(mdl6,test_inputs);
%
%
%
% %mdl7 = fitlm(X_z_train,Y_z_train(:,7));
% mdl7 = fitrlinear(train_inputs,train_labels(:,7));
% ypred7 = predict(mdl7,test_inputs);
%
%
% %mdl8 = fitlm(X_z_train,Y_z_train(:,8));
% mdl8 = fitrlinear(train_inputs,train_labels(:,8));
% ypred8 = predict(mdl8,test_inputs);
%
%
% %mdl9 = fitlm(X_z_train,Y_z_train(:,9));
% mdl9 = fitrlinear(train_inputs,train_labels(:,9));
% ypred9 = predict(mdl9,test_inputs);
%
%
% pred_labels = [ypred1 ypred2 ypred3 ypred4 ypred5 ypred6 ypred7 ypred8 ypred9];



end


