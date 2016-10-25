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
function accuracy = earlyFusionBatch(seed, CodeNum, K, knn, sub, threshold, dim, alpha)


%-------------------------- Random Parameters -----------------------------
rng(seed);
nClass=6;

timeDuration=15;
dimension = 6;

trainPath = ['..\dataset\HAPT\SCV\trainset_PCA_' num2str(sub) '.mat'];
testPath = ['..\dataset\HAPT\SCV\testset_PCA_' num2str(sub) '.mat'];
load(trainPath);
load(testPath);

trainfeat = trainset.feats;
testfeat = testset.feats;
trainlabel = trainset.class;
testlabel = testset.class;
numTrain = size(trainfeat,1);
numTest = size(testfeat,1);

superHashCodeTest = [];
superHashCode=[];
%% mean 
%dim = 35;
%threshold = 5.
trainset.feats = [trainfeat(:,0*dimension+1:0*dimension+dim,:) -1*ones(numTrain,1)];
testset.feats = [testfeat(:,0*dimension+1:0*dimension+dim,:), -1*ones(numTest,1)];  


fileName = ['..\dataset\HAPT\noBoosting\Learned_feat1_L' ...
    num2str(CodeNum) '_K' num2str(K) '_sub' num2str(sub) '.mat'];
load(fileName);
start1=tic;                                                 
[SampleHashCodes, TestSampleCodes]= ...
                    FTAHash(learnedProjection, trainset, testset, ...
                          CodeNum, K, threshold(1), timeDuration, alpha);
elapse1 = toc(start1);

superHashCode = [superHashCode SampleHashCodes];
superHashCodeTest = [superHashCodeTest TestSampleCodes];
%HammingDistanceArrayMean = pdist2(TestSampleCodes, SampleHashCodes, 'hamming'); 
%HammingDistanceArrayMean = ComputeHammingDist(TestSampleCodes, SampleHashCodes);

%% min
   
trainset.feats = [trainfeat(:,1*dimension+1:1*dimension+dim) -1*ones(numTrain,1)];
testset.feats = [testfeat(:,1*dimension+1:1*dimension+dim) -1*ones(numTest,1)];     


fileName = ['..\dataset\HAPT\noBoosting\Learned_feat2_L' ...
    num2str(CodeNum) '_K' num2str(K) '_sub' num2str(sub) '.mat'];
load(fileName);
start2 = tic;                                                
[SampleHashCodes, TestSampleCodes]= ...
                    FTAHash(learnedProjection, trainset, testset, ...
                          CodeNum, K, threshold(2), timeDuration, alpha);
elapse2 = toc(start2);
%HammingDistanceArrayMin = pdist2(TestSampleCodes, SampleHashCodes, 'hamming'); 
superHashCode = [superHashCode SampleHashCodes];
superHashCodeTest = [superHashCodeTest TestSampleCodes];

%% max
%threshold = 3 
trainset.feats = [trainfeat(:,2*dimension+1:2*dimension+dim) -1*ones(numTrain,1)];
testset.feats = [testfeat(:,2*dimension+1:2*dimension+dim) -1*ones(numTest,1)];  


fileName = ['..\dataset\HAPT\noBoosting\Learned_feat3_L' ...
    num2str(CodeNum) '_K' num2str(K) '_sub' num2str(sub) '.mat'];
load(fileName);
start3 = tic;                                                
[SampleHashCodes, TestSampleCodes]= ...
                    FTAHash(learnedProjection, trainset, testset, ...
                          CodeNum, K, threshold(3), timeDuration, alpha);
elapse3 = toc(start3);
superHashCode = [superHashCode SampleHashCodes];
superHashCodeTest = [superHashCodeTest TestSampleCodes];
%HammingDistanceArrayMax = pdist2(TestSampleCodes, SampleHashCodes, 'hamming');

%% variance
%threshold = 3
trainset.feats = [trainfeat(:,3*dimension+1:3*dimension+dim) -1*ones(numTrain,1)];
testset.feats = [testfeat(:,3*dimension+1:3*dimension+dim) -1*ones(numTest,1)];     

fileName = ['..\dataset\HAPT\noBoosting\Learned_feat4_L' ...
    num2str(CodeNum) '_K' num2str(K) '_sub' num2str(sub) '.mat'];

load(fileName);
 start4 = tic;                                                
[SampleHashCodes, TestSampleCodes]= ...
                    FTAHash(learnedProjection, trainset, testset, ...
                          CodeNum, K, threshold(4), timeDuration, alpha);
elapse4 = toc(start4);
superHashCode = [superHashCode SampleHashCodes];
superHashCodeTest = [superHashCodeTest TestSampleCodes];
%HammingDistanceArrayVar = pdist2(TestSampleCodes, SampleHashCodes, 'hamming');

%% late fusion
HammingDistanceArray = pdist2(superHashCodeTest, superHashCode, 'hamming');

accuracy = computeAccuray(HammingDistanceArray, testlabel,...
                                                trainlabel, knn, nClass);         
% fprintf('accuracy = %.4f\n', accuracy);

%fprintf('time = %.4f second\n', elapse1+elapse2+elapse3+elapse4);

end


