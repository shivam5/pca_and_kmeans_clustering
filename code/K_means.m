function [Acc, Conf_Mat, Label_C] = K_means(X, Y, K)
%   K_MEANS runs K-means clusstering algorithm on 
%   given data matrix X, assigns most frequently 
%   occurring label to each cluster and then
%   computes accuracy and constructs confusion matrix

%% Initializing useful parameters

% Number of examples
N = size(X,1);

%% Performing K-means Clusterring

[idx] = kmeans(X, K);

%% Assigning most frequent label to each cluster

Label_C = zeros(K,1);
for k = 1:K
    count = zeros(1,10);
    for i = 1:N
        if(idx(i,1) == k)
            label = Y(i) + 1;
            count(1,label) = count(1,label) + 1;
        end
    end
    maximum = -1;
    index = -1;
    for i = 1:10
        if(count(1,i) > maximum)
            maximum = count(1,i);
            index = i;
        end
    end
    Label_C(k,1) = index-1;
end

%% Predicting Labels

Predicted = zeros(N,1);
N_correct = 0;
for i = 1:N
    Predicted(i,1) = Label_C(idx(i),1);
    if(Predicted(i,1) == Y(i,1))
        N_correct = N_correct + 1;
    end
end

%% Computing Accuracy

Acc = (N_correct/N) * 100;

%% Designing Confusion Matrix

Conf_Mat = zeros(10,10);

for i = 1:N
    true_label = Y(i,1);
    predicted_label = Predicted(i,1);
    Conf_Mat(true_label+1, predicted_label+1) = Conf_Mat(true_label+1, predicted_label+1) + 1; 
end

end
