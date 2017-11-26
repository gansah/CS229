%% SVM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% svm_train.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rand('seed', 123);

Xtrain = year1(2027:6956, 1:64);
m_train = size(Xtrain, 1);
ytrain = year1(2027:6956, end);
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

Xtest = year1([1:2027, 6957:end], 1:64);
ytest = year1([1:2027, 6957:end], end);
ytest = (2 * ytest - 1);

% Xtest = year5(:, 1:64);
% ytest = year5(:,65);
% ytest = (2 * ytest - 1);

% Construct test and train matrices
Xtest = 1.0 * (Xtest > 0);
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

