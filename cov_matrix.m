clc; clear;
load 'years' % load year1, year2, year3, year4, year5

% Using year; can change the year 
year = year3;
Xmatrix = year(:, 1: size(year, 2)-1);

% Finding the mean for each feature
sample_mean = mean(Xmatrix);

% Finding the sample variance for each feature
sample_var = var(Xmatrix);

% Cov Matrix
sample_cov_matrix = cov(Xmatrix);

% Correlation Matrix
sample_cor_matrix = corrcoef(Xmatrix);

% Correlation between feature and output
% R = corrcoef(year3);

%%
alt_cor = sample_cor_matrix;
for i = 1:64
    for j = 1:64
        %if -0.3 < correlation(i,j) < 0.3 
        if  abs(alt_cor(i,j)) < 0.1  
            alt_cor(i,j) = 0;
        else 
            alt_cor(i,j) = 1;
        end
    end
end
%%
[row, col] = find(alt_cor == 1);
ind = [col row];

%% Lucio something something 

[hoho score latent] = pca(Xmatrix, 'NumComponents', 5);

%plot(score(:,1), score(:,2))
scatter(score(:,1), score(:,2))
y = year3(:,end);
ind0 = find(ismember(y,0)); ind1 = find(ismember(y,1));
scatter(score(ind0,1), score(ind0,2))
hold on
scatter(score(ind1,1), score(ind1,2))
legend('ind0', 'ind1')

%%
 hoho = pca(Xmatrix);
[hoho scorehoho latent] = pca(Xmatrix);
plot(scorehoho(:,1), scorehoho(:,2))
scatter(scorehoho(:,1), scorehoho(:,2))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = year3(:,end);
ind0 = find(ismember(y,0)); ind1 = find(ismember(y,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[hhoho sscore llatent] = pca(Xmatrix, 'NumComponents', 2);
scatter(sscore(ind0,1), sscore(ind0,2))
hold on
scatter(sscore(ind1,1), sscore(ind1,2))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[hhoho sscore llatent] = pca(Xmatrix, 'NumComponents', 3);
scatter3(sscore(ind0,1), sscore(ind0,2), sscore(ind0,3))
hold on
scatter3(sscore(ind1,1), sscore(ind1,2), sscore(ind1,3))
