%% Attempting Naive Bayes

trainMatrix = year1(2027:6956, 1:64);
ytrain = year1(2027:6956, end);

testMatrix = year1([1:2027, 6957:end], 1:64);
ytest = year1([1:2027, 6957:end], end);

numTokens = 64;
numTestDocs = length(ytest);
numTrainDocs = length(ytrain);

phi_y1 = zeros(1, 64);
phi_y0 = zeros(1, 64);

spam_Matrix = zeros(1, numTokens);
nspam_Matrix = zeros(1, numTokens);



for i = 1:numTrainDocs
    if(ytrain(i) == 1)
        spam_Matrix = vertcat(spam_Matrix, trainMatrix(i,:));
    else
        nspam_Matrix = vertcat(nspam_Matrix, trainMatrix(i,:));
    end
end
spam_Matrix = spam_Matrix(2:end, :);
nspam_Matrix = nspam_Matrix(2:end, :);

spam_sum = sum(spam_Matrix);
nspam_sum = sum(nspam_Matrix);


for j = 1:numTokens
    phi_y1(1, j) = (spam_sum(j) + 1)/(sum(spam_sum) + numTokens);
    phi_y0(1, j) = (nspam_sum(j) + 1)/(sum(nspam_sum) + numTokens);
end
phi_y = size(spam_Matrix, 1)/numTrainDocs;


% Testing
output = zeros(numTestDocs, 1);

for i = 1:numTestDocs
    prob_y1_x = 0; prob_y0_x = 0;
    for j = 1:numTokens
        prob_y1_x = prob_y1_x + testMatrix(i, j)*log(phi_y1(j));
        prob_y0_x = prob_y0_x + testMatrix(i, j)*log(phi_y0(j));
    end
    prob_y1_x = prob_y1_x * phi_y; prob_y0_x = prob_y0_x * phi_y;
    
    if(prob_y1_x > prob_y0_x)
        output(i, 1) = 1;
    elseif prob_y1_x < prob_y0_x
        output(i, 1) = 0;
    elseif prob_y1_x == prob_y0_x
        output(i, 1) = randi([0 1]);
    end
end

% Compute the error on the test set
y = full(ytest);
y = y(:);
NBerror = sum(y ~= output) / numTestDocs;