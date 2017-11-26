%% Getting data
load 'years'

%% Training data 
trainX = year1(2027:6956, 1:64);
trainY = year1(2027:6956, end);
temp_var = ones(size(trainX,1), 1);
trainX = horzcat(temp_var, trainX);

est_param = inv(transpose(trainX)*trainX) * transpose(trainX)*trainY;
%% Training error 
est_y = trainX * est_param;
k_1 = find(est_y < 0.5); k_2 = find(est_y > 0.5); 
copyY = est_y; copyY(k_1) = 0; copyY(k_2) = 1;
error_index = find(copyY == trainY);
error = (length(trainY) - length(error_index))/(length(trainY));

%% Test Error
testX = year1([1:2027, 6957:end], 1:64); 
testY = year1([1:2027, 6957:end], end);
temp_var = ones(size(testX,1), 1); testX = horzcat(temp_var, testX);
test_y = testX * est_param;
k_1 = find(test_y < 0.5); k_2 = find(test_y > 0.5); 
copytestY = test_y; copytestY(k_1) = 0; copytestY(k_2) = 1;
test_error_index = find(copytestY == testY);
test_error = (length(testY) - length(test_error_index))/(length(testY));

%% test error for year 2
errors = zeros(1, length(year));
for i = 1:length(year)
    
    temp_var = ones(size(year5,1), 1); year2X = horzcat(temp_var, year5(:, 1:64));
    year2Y = year5(:, 65);
    test_y2 = year2X * est_param;
    k_1 = find(test_y2 < 0.5); k_2 = find(test_y2 > 0.5);
    copytestY = test_y2; copytestY(k_1) = 0; copytestY(k_2) = 1;
    test_error_index = find(copytestY == year2Y);
    test_error_y2 = (length(year2Y) - length(test_error_index))/(length(year2Y));
end