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

% main for the FTAhashing
% make sure to clear the working space before proceeding
function accuracy  = lateFusionBatch(seed, CodeNum, K, knn, sub, threshold, dim, alpha)


%-------------------------- Random Parameters -----------------------------
rng(seed);
nClass=6;

timeDuration=15;

trainPath = ['..\dataset\HAPT\SCV\trainset_PCA_' num2str(sub) '.mat'];
testPath = ['..\dataset\HAPT\SCV\testset_PCA_' num2str(sub) '.mat'];
load(trainPath);
load(testPath);

numTrain = size(trainset.feats,1);
numTest = size(testset.feats,1);

numFeatures = 4;

%% mean 
%threshold = 5.
trainset.feats = [trainset.feats(:,1:numFeatures*dim,:) -1*ones(numTrain,1)];
testset.feats = [testset.feats(:,1:numFeatures*dim,:), -1*ones(numTest,1)];  


fileName = ...
['..\dataset\HAPT\noBoosting\Learned_earlyfuse_thre_' num2str(threshold) 'L' ...
    num2str(CodeNum) '_K' num2str(K) '_sub' num2str(sub) '.mat'];
load(fileName);
start1=tic;                                                 
[SampleHashCodes, TestSampleCodes]= ...
                    FTAHash(learnedProjection, trainset, testset, ...
                          CodeNum, K, threshold, timeDuration, alpha);
elapse1 = toc(start1);

start2 = tic;

HammingDistanceArray = pdist2(TestSampleCodes, SampleHashCodes, 'hamming'); 

accuracy = computeAccuraySingle(HammingDistanceArray, testset.class,...
                                                trainset.class, knn, nClass); 
elapse2 = toc(start2);

fprintf('pFTA feature extraction time = %f, knn classification time = %f\n', ...
                                                        elapse1, elapse2);

end


