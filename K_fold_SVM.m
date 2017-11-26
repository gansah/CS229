clc; clear;
load 'years'
X1 = year3(:, 1:end-1); Y1 = year3(:,end); % Can change the year
k = 20; SVMerror = zeros(1, k);
index1 = find(ismember(Y1, 0)); index2 = find(ismember(Y1, 1));
X1_nbank = X1(index1, :); X1_bank = X1(index2, :);
Y1_nbank = Y1(index1); Y1_bank = Y1(index2);
nbank_s = size(Y1_nbank); bank_s = size(Y1_bank);
temp_v1 = floor(nbank_s/k); temp_v2 = floor(bank_s/k);

%SavedFloor = floor(size(Matrix,1)/10)
s1 = 1; s2 = 1;
for i = 1:k
    copy_X1_nbank = X1_nbank; copy_X1_bank = X1_bank;
    copy_Y1_nbank = Y1_nbank; copy_Y1_bank = Y1_bank;
    
    if i ~= k
        index1 = s1:(temp_v1*i); index2 = s2:(temp_v2*i);
        
        X_kb = X1_bank(index2, :); Yk_b = Y1_bank(index2);
        X_knb = X1_nbank(index1, :); Y_knb = Y1_nbank(index1);
        
        copy_X1_nbank(index1, :) = []; copy_Y1_nbank(index1) = [];
        copy_X1_bank(index2, :) = []; copy_Y1_bank(index2) = [];
        
    else
        X_kb = X1_bank(s2:end, :); Yk_b = Y1_bank(s2:end);
        X_knb = X1_nbank(s1:end, :); Y_knb = Y1_nbank(s1:end);
        
        copy_X1_nbank(s1:end, :) = []; copy_Y1_nbank(s1:end) = [];
        copy_X1_bank(s2:end, :) = []; copy_Y1_bank(s2:end) = [];
    end
    s1 = temp_v1*i + 1; s2 = temp_v2*i + 1;
    
    Xtrain = vertcat(copy_X1_nbank, copy_X1_bank);
    ytrain = vertcat(copy_Y1_nbank, copy_Y1_bank);
    
    % Xtest = vertcat(X_knb ,X_kb); ytest = vertcat(Y_knb, Yk_b);
    Xtest = Xtrain; ytest = ytrain;
    rand('seed', 123);
    
    m_train = size(Xtrain, 1);
    ytrain = (2 * ytrain - 1);
    Xtrain = 1.0 * (Xtrain > 0);
    squared_X_train = sum(Xtrain.^2, 2);
    gram_train = Xtrain * Xtrain';
    tau = 8;
    
    % Get full training matrix for kernels using vectorized code.
    Ktrain = full(exp(-(repmat(squared_X_train, 1, m_train) ...
        + repmat(squared_X_train', m_train, 1) ...
        - 2 * gram_train) / (2 * tau^2)));
    
    lambda = 1 / (64 * m_train);
    num_outer_loops = 40;
    alpha = zeros(m_train, 1);
    
    avg_alpha = zeros(m_train, 1);
    Imat = eye(m_train);
    
    count = 0;
    for ii = 1:(num_outer_loops * m_train)
        count = count + 1;
        ind = ceil(rand * m_train);
        margin = ytrain(ind) * Ktrain(ind, :) * alpha;
        g = -(margin < 1) * ytrain(ind) * Ktrain(:, ind) + ...
            m_train * lambda * (Ktrain(:, ind) * alpha(ind));
        % g(ind) = g(ind) + m_train * lambda * Ktrain(ind,:) * alpha;
        alpha = alpha - g / sqrt(count);
        avg_alpha = avg_alpha + alpha;
    end
    avg_alpha = avg_alpha / (num_outer_loops * m_train);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % svm_test.m
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Xtest = 1.0 * (Xtest > 0);
    ytest = (2 * ytest - 1);
    squared_X_test = sum(Xtest.^2, 2);
    m_test = size(Xtest, 1);
    gram_test = Xtest * Xtrain';
    Ktest = full(exp(-(repmat(squared_X_test, 1, m_train) ...
        + repmat(squared_X_train', m_test, 1) ...
        - 2 * gram_test) / (2 * tau^2)));
    
    % preds = Ktest * alpha;
    
    % fprintf(1, 'Test error rate for final alpha: %1.4f\n', ...
    %         sum(preds .* ytest <= 0) / length(ytest));
    
    preds = Ktest * avg_alpha;
    fprintf(1, 'Test error rate for average alpha: %1.4f\n', ...
        sum(preds .* ytest <= 0) / length(ytest));
    test_errorSVM = sum(preds .* ytest <= 0) / length(ytest);
    SVMerror(i) = test_errorSVM; 
end