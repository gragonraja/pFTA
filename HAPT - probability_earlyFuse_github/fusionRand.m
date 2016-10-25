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


function accuracy = fusionRand(seed, K, dim, CodeNum, sub, threshold, knn, alpha)

% main for the FTAhashing
% make sure to clear the working space before proceeding

%-------------------------- Random Parameters -----------------------------
rng(seed);
nClass=6;
timeDuration=15;
numFeatures = 4;

trainPath = ['..\dataset\HAPT\SCV\trainset_PCA_' num2str(sub) '.mat'];
testPath=['..\dataset\HAPT\SCV\testset_PCA_' num2str(sub) '.mat'];
load(trainPath);
load(testPath);

numTrain = size(trainset.feats,1);
numTest = size(testset.feats,1);
tstart = tic;
%% mean 
trainset.feats = [trainset.feats(:,1:numFeatures*dim), -1*ones(numTrain,1)];
testset.feats = [testset.feats(:,1:numFeatures*dim), -1*ones(numTest,1)];  

normalizedProjection = [GaussianRandomGenerator(dim*numFeatures, CodeNum*K) ...
                                                threshold*ones(K*CodeNum,1)];   

%fileName = ['../dataset/HAPT/rand/rand_feat1_' num2str(sub)];
%load(fileName);
%save(fileName, 'normalizedProjection', '-v7.3');
[SampleHashCodes, TestSampleCodes]= ...
                    FTAHash(normalizedProjection, trainset, testset, ...
                          CodeNum, K, -inf, timeDuration, alpha);


HammingDistanceArray = pdist2(TestSampleCodes, SampleHashCodes, 'hamming');                      

accuracy = computeAccuray(HammingDistanceArray, testset.class,...
                                                trainset.class, knn, nClass);                                    
elapse = toc(tstart);
%fprintf('time elapse = %.4f\n', elapse);
end



