function [X_transformed, reconst_error, N_Dim, U_reduce, Reconst_X] = PCA (X, choice, N_Dim)
%   PCA performs dimeensionality reduction on original data
%   if choice is 1 then it performs PCA on given number of 
%   dimensions otherwise it finds appropriate value of dimensions
%   upon which projecting data results in reconstruction error < 0.1

%% Initializing useful parameters

% Number of Examples
N = size(X,1);

% Dimension of Data
D = size(X,2);

%% Normalizing the Input Matrix

train_X = X;

mu = mean(X);
for i = 1:N
    X(i,:) = X(i,:) - mu;
end

%% Performing PCA

% Calculating Covariance Matrix
Sigma = (X' * X)/N;

% Finding eigen vectors and eigen values of Sigma
[U, S] = eig(Sigma);
    
% Singular value decomposition of Sigma
%[U, S, V] = svd(Sigma);

% Finding appropriated number of dimensions to project
if(choice == 1)
    % Finding k largest eigen values
    eig_vals = zeros(D,1);
    for i = 1:D
        eig_vals(i,1) = S(i,i);
    end
    [val, vector] = sortrows(eig_vals);
    % Finding k vectors corresponding to eigen values
    U_reduce = zeros(D,N_Dim);
    for i = 1:N_Dim
        index = vector(D-i+1,1);
        U_reduce(:,i) = U(:,index);
    end
    X_transformed = X * U_reduce;
    % Reconstructing Data Matrix
    Reconst_X = X_transformed * U_reduce';
    % Adding mean
    for i = 1:N
        Reconst_X(i,:) = Reconst_X(i,:) + mu;
    end
    % Computing Reconstruction Error
    reconst_error = 0;
    for i = 1:N
        reconst_error = reconst_error + ( (train_X(i,:) - Reconst_X(i,:)) * (train_X(i,:) - Reconst_X(i,:))' ); 
    end
    reconst_error = reconst_error / N;

elseif(choice == 2)
    for i = 1:D
        N_Dim = i;
        % Finding k largest eigen values
        eig_vals = zeros(D,1);
        for j = 1:D
            eig_vals(j,1) = S(j,j);
        end
        [val, vector] = sortrows(eig_vals);
        % Finding k vectors corresponding to eigen values
        U_reduce = zeros(D,N_Dim);
        for j = 1:N_Dim
            index = vector(D-j+1,1);
            U_reduce(:,j) = U(:,index);
        end
        
        %U_reduce = U(:, 1:N_Dim);
        X_transformed = X * U_reduce;
        % Reconstructing Data Matrix
        Reconst_X = X_transformed * U_reduce';
        % Adding mean
        for j = 1:N
            Reconst_X(j,:) = Reconst_X(j,:) + mu;
        end
        % Computing Reconstruction Error
        reconst_error = 0;
        for j = 1:N
            reconst_error = reconst_error + ( (train_X(j,:) - Reconst_X(j,:)) * (train_X(j,:) - Reconst_X(j,:))' ); 
        end
        reconst_error = reconst_error / N;
        if(reconst_error < 0.1)
            break;
        end
    end
end
    
end