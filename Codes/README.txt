STRUCTURE OF OUR SUBMISSION:

All of our work is in Submissions folder of which this README.txt file is a part of. You would find 4 different folders, one for each method: 


1. Generative     : This folder has a k_means clustering method, with predictions being made with a linear model
2. Discriminative : This folder has a linear regression model in linearfit.m and a Cross_validation folder 
3. Instance       : This folder has a Gaussian kernel SVM regression model
4. Novel          : This folder has two models, one each in the folders:  First, novel.m executes a method that combines all of our previous methods using a simple average. This includes one new model, a partial least squares regression. The PLS model is also available in a separate folder.









A GENERATIVE METHOD:


K-means Clustering

The purpose of this method is to identify similar features and group them as a single feature to reduce redundancy and prevent overfitting.

k_means_main.m: Entry point for performing k_means method on training and test data. Simply executing the function will import the files and carry out 5-fold cross validation for a range of K. The program outputs the ideal k and the resulting crossvalidation error using the provided error metric.

k_means_method.m: Performs k_mean_clustering and performs fitrlinear using default parameters on the dataset. Returns a prediction column which can be used to calculate error.

These files also rely on the provided "error_metric.m" file and the "training_data.mat" file.







A DISCRIMINATIVE METHOD:

Here we use linear regression model with ridge regularization as an example case. Though this is not the part of the code we submitted to leaderboard. 

linearfit.m 
Run this code to get an estimate of the cost it gives after the ridge regularization parameters are tuned from cross validation. You would get the output for training and testing error.


error_metric.m
This is the sub module that you have provided us with to compute the cost of our models.


Cross_validation
Please enter this folder if you are interested to know how we perform cross validation. Run the code cross_validation.m. The code is self sufficient and would give you a plot of cross-validation error Vs Lambda (the ridge regularization penalty parameter).

Inside the Cross_validation folder:

Run the cross_validation.m code. We understand that the optimal lambda for each response column in Y might vary and hence we tackle each column separately. We assign the vector lambda with values in orders of 10, and then zoom in the values which show minimum cross validation error to get much better estimate of the optimal lambda values.




AN INSTANCE BASED METHOD:

Gaussian Kernel SVM
This method uses support vector machines with a gaussian kernel to predict y_labels on the dataset.

svm_main.m: This is the file that should be executed to obtain both the cv_error and the test error on the svm regression on the full dataset. It will print the cross-validation error and training error to the console.

svm_predict.m: This contains the general predict_labels method that will return a set of predictions if fed an Xtrain, Ytrain, and Xtest.

These files also rely on the provided "error_metric.m" file and the "training_data.mat" file being present.







A NOVEL METHOD:

We implement an averaging technique on Partial least Squares Regression and SVM regression. These two are uncorrelated methods and have uncorrelated errors and hence on doing an average the errors  look like noise and hence cancel out. Also, the ensemble often fits a much richer class of functions than any single method (e.g. a decision tree is relatively crude set of models compared to a random forest composed of those same decision trees).

Partial Least Square Regression:

PLS regression has a tuning parameter which is the number of components to be considered (k). We tune this parameter for each Y -response.

plsrfit.m
Run this code to get an estimate of the cost of this model after tuning from cross validation. You would get the output for training and testing error.

error_metric.m
This is the sub module that you have provided us with to compute the cost of our models.

Cross_Validation
Please enter this folder if you are interested to know how we compute the number of principal components(k). Run cross_validation.m in this folder to have a better idea. 








SVM:


error_metric.m
This is the sub module that you have provided us with to compute the cost of our models.