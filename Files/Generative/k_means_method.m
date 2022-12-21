function  pred_labels = k_means_method(train_inputs, train_labels, test_inputs, k_val)
    %%Log transform of twitter data
    train_inputs(:,22:end) = log(train_inputs(:,22:end));
    test_inputs(:,22:end) = log (test_inputs(:,22:end));
    %%Normalization of all data, both test and train
    [Z_train,mu_train,sigma_train] = zscore(train_inputs,0,1);
    Z_test = (test_inputs-mu_train)./sigma_train;
    [Z_freqs mu_freqs sigma_freqs] = zscore(train_inputs(:, 22:end));
    [Z_freqs2 mu_freqs2 sigma_freqs2] = zscore(test_inputs(:, 22:end));
    %%Do k_means for training data
    idx = kmeans(Z_freqs', k_val)'; %Get K_means index labels
    kmzf = zeros(size(Z_freqs, 1), max(max(idx))); %Create empty matrix to hold data
    for ii = 1:max(max(idx)) %loop through all k_means indexes
        mat = Z_freqs(:,idx == ii); %get all columns that match that index
        kmzf(:, ii) = mean(mat,2); %average them
    end
    kmzf = [Z_train(:,1:21), kmzf]; %append averaged values to regional data
    %%Repeat above for testing data
    kmzf2 = zeros(size(Z_freqs2, 1), max(max(idx)));
    for ii = 1:max(max(idx))
        mat2 = Z_freqs2(:,idx == ii);
        kmzf2(:, ii) = mean(mat2,2);
    end
    kmzf2 = [Z_test(:,1:21), kmzf2];

    %%Train a linear model on data and return results
    for ii = 1:9
        model = fitrlinear(kmzf, train_labels(:,ii));
        pred_labels(:,ii) = predict(model, kmzf2);
    end

end