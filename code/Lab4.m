%% ==================== CSL 603, Machine Learning - K-Means Clusstering and PCA ====================

%% Initialization
clear; close all; clc;

%% Loading Data and Labels

% X - Data Matrix
% Y - Label Matrix

X = load('../data.txt');

% Number of examples
N = size(X,1);

% Dimension of data
D = size(X, 2);

Y_temp = load('../label.txt');

% Extracting Actual label
Y = zeros(N,1);
for i = 1:N
    for j = 1:10
        if(Y_temp(i,j) == 1)
            if(j == 10)
                Y(i,1) = 0;
            else
                Y(i,1) = j;
            end
        end
    end
end

%% Running K-Means on original Dimensions

% Number of Clusters
K = input('Enter number of clusters required for K-means clusterring - ');

fprintf('\nPerforming K-Means Clusstering with K = %d\n\n', K);

[Acc, Conf_Mat, Label_C] = K_means(X,Y,K);

% Displaying Accuracy
fprintf('Accuracy = %f\n\n', Acc);

% Displaying Cluster Labels
fprintf('Cluster Labels - \n');
disp(Label_C);

% Displaying Confusion Matrix
fprintf('Confusion Matrix -\n\n');
disp(Conf_Mat);

pause(5);
%% Performing PCA on original Data

fprintf('1. Do you want to provide Number of Dimensions to project data\n');
fprintf('2. Do you want to find appropriate dimensions through which error < 0.1\n\n');

choice = input('Enter your choice 1 or 2 - ');

N_dim = 0;
if(choice == 1)
    N_dim = input('Enter number of dimensions on which Original Data is to be projected - ');
    fprintf('\nPerforming PCA to project data to %d dimensions -\n\n', N_dim);

elseif(choice == 2)
    fprintf('\nPerforming PCA and finding appropriate number of dimensions\n\n');
end

[X_reduced, reconst_error, N_dim, U] = PCA(X, choice, N_dim);

if(choice == 2)
    fprintf('Appropriate Number of Dimensions with error < 0.1  = %d\n\n', N_dim);
end

% Displaying Reconstuction Error
fprintf('Reconstruction Error = %f\n\n', reconst_error);

pause(5);
%% Running K-Means on reduced Dimensions

% Number of Clusters
K = input('Enter number of clussters required for K-means clusterring - ');

fprintf('\nPerforming K-Means Clusstering with K = %d\n\n', K);

[Acc, Conf_Mat, Label_C] = K_means(X_reduced,Y,K);

% Displaying Accuracy
fprintf('Accuracy = %f\n\n', Acc);

% Displaying Cluster Labels
fprintf('Cluster Labels - \n');
disp(Label_C);

% Displaying Confusion Matrix
fprintf('Confusion Matrix -\n');
disp(Conf_Mat);

