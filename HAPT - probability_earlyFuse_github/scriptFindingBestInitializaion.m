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
addpath '../mtimesx';
seed = 'default';
rng(seed);
dim = 6;
numFeatures = 4;
CodeNum = 16;
K = 22;
nClass = 6;
alpha = 5;
num=6; % num of subjects in each group
maxKnn = 30;  % max num of knn used for cross validation

for sub=1:5
    fprintf('=========\nprocessing sub %d, L=%d, K=%d\n', sub, CodeNum, K);
   
    trainPath = ['..\dataset\HAPT\SCV\trainset_PCA_' num2str(sub) '.mat'];
    load(trainPath);
    numTrain = size(trainset.feats,1);
    trainset.feats = [trainset.feats(:,1:numFeatures*dim,:) -1*ones(numTrain,1)];

    for threshold = 1.2:0.1:2.2
        % random projection
        learnedProjection = [GaussianRandomGenerator(dim*numFeatures, CodeNum*K) ...
                                            threshold*ones(K*CodeNum,1)];   

        accuracyAll = getCrossValidationKNN(sub, num, trainset, ...
            learnedProjection, CodeNum, K, threshold, nClass, alpha, maxKnn);
        meanAcc = mean(accuracyAll,2);
        [bestAccuracy, bestKnn]=max(meanAcc);
        fprintf('sub %d, threshold = %.4f, best accuracy %.4f @ knn = %d\n', ...
            sub, threshold, bestAccuracy,  bestKnn);
    end
end

