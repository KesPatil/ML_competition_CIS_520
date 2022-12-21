clear all;
close all;
%Import Data
load training_data.mat; %outputs train_inputs and train_labels
part = make_xval_partition(size(train_inputs, 1), 5)';
error = xval_error(train_inputs, train_labels, 0.01, part)

ypred1 = predict_labels(train_inputs,train_labels,train_inputs);
training_error  = error_metric(ypred1, train_labels)

% error_min = 10000000;
% min_params = [0 0 0];
% for jj = 1:100000
%     rng = rand([1 3]);
%     rng = rng./sum(rng);
%     for ii = 1:5
%         cur_data = all_data(:, 4*(ii-1)+1:4*(ii));
%         eval = sum(rng.*cur_data(:, 1:3),2);
%         error(ii) = sum(eval - cur_data(4));
%     end
%     total_error = mean(error);
%     if(total_error < error_min) 
%         error_min = total_error;
%         min_params = rng;
%     end
% end