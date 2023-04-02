function [testerror,nofeatures] = EvalFunction(dataTrain, classTrain, dataTest, classTest, hyperparam)
% Training a model ("hyperparam.method") on training data and evaluation based on the test set

% Hyperparameters contained in "hyperparam" input

if hyperparam.method == 'KNN' 
    %% KNN classifier 
    
    % Model Training (SPECIAL since lazy learner)
    mdl = fitcknn(dataTrain,classTrain,'Distance','euclidean','NumNeighbors',hyperparam.methodKNNkval);
    
    % Test Error
    testerror = 1 - mean(predict(mdl,dataTest)==classTest);
    
    % Number of Features
    nofeatures = size(dataTest,2);   
else
    
end

end

