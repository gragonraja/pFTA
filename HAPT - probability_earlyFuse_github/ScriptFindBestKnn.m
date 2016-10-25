% Copyright (c) Jun Ye. 2016. This program is free software: you can 
% redistribute it and/or modify it under the terms of the 
% GNU General Public License as published by the Free Software Foundation, 
% either version 3 of the License, or (at your option) any later version. 
% This program is distributed in the hope that it will be useful, 
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
% 
% See the GNU General Public License for more details. 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see http://www.gnu.org/licenses/.

clear all;
seed = 'default';
addpath '../mtimesx';
rng(seed);
dim = 6;
numFeatures = 4;
CodeNum = 32;
K = 22;
nClass = 6;
alpha = 5;
num=6; % num of subjects in each group
maxKnn = 15;  % max num of knn used for cross validation

threshold =[1.9 2.1 2.1 1.9 1.9];

for sub = 1:5
    trainPath = ['..\dataset\HAPT\SCV\trainset_PCA_' num2str(sub) '.mat'];
    load(trainPath);
    numTrain = size(trainset.feats,1);
    trainset.feats = [trainset.feats(:,1:numFeatures*dim,:) -1*ones(numTrain,1)];

    % load learned projection
    fileName = ['..\dataset\HAPT\noBoosting\Learned_earlyfuse_thre_' num2str(threshold(sub)) 'L' ...
       num2str(CodeNum) '_K' num2str(K) '_sub' num2str(sub) '.mat'];
    load(fileName);
    
    % random projection
%     learnedProjection = ...
%         [GaussianRandomGenerator(dim*numFeatures, CodeNum*K) ...
%         threshold*ones(K*CodeNum,1)];   

    
    accuracyAll = getCrossValidationKNN(sub, num, trainset, learnedProjection,...
        CodeNum, K, threshold(sub), nClass, alpha, maxKnn);

    
    meanAcc = mean(accuracyAll,2);
    [bestAccuracy, bestKnn]=max(meanAcc);
    fprintf('sub %d, K=%d, L=%d, best accuracy %.4f @ knn = %d\n', sub, K, ...
                                            CodeNum, bestAccuracy,  bestKnn);
end

