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

function LearnScriptNoBoosting(seed, dim, CodeNum, K, sub, ...
    learningRate, lambda1, alpha1, MaxEpoch, threshold)

%-------------------------- Random Parameters -----------------------------
rng(seed);
nClass=6;
dimension = 6; % feature dimension
timeDuration=15;

%------------------------ Learning Parameters -----------------------------
%lambda1 = 0.001; % L2 regularizer 0.001
lambda2 = 0.5; % momentum 0.5
%threshold = -inf;    

sampleRate = 0.1; 
positivePairSampleRate =  0.1;
negativePairSampleRate = 0.1;%0.005;
numFeature = 4;
%% display parameters
fprintf('===============settings===============\n');
fprintf('pcadim = %d, K=%d, codeLen=%d,\n', dim, K, CodeNum);
fprintf('LR=%f, thres=%.3f, epoch = %d\n', learningRate, threshold, MaxEpoch);
fprintf('SR = %.3f, pPairRate = %.3f, nPairRate=%.3f\n', sampleRate,...
                         positivePairSampleRate, negativePairSampleRate);
fprintf('alpha = %f\n', alpha1);
fprintf('lambda1 = %f, lambda2 = %f\n', lambda1, lambda2 );

%% load datasets
trainPath = ['..\dataset\HAPT\SCV\trainset_PCA_' num2str(sub) '.mat']
load(trainPath);

numTrain = size(trainset.feats,1);
trainset.feats = [trainset.feats(:,1:numFeature*dimension) -1*ones(numTrain,1)];

%% Random
normalizedProjection = [GaussianRandomGenerator(dim*numFeature, CodeNum*K) threshold*ones(K*CodeNum,1)];  
  
%  fileName = ['../dataset/HAPT/rand/rand_feat' num2str(featIndex) '_' num2str(sub)];
%  load(fileName);

%% FTA Learning
tstart=tic;
learnedProjection = zeros(K*CodeNum, dim*numFeature+1);

for c=1:CodeNum
    fprintf('Learning code %d: ', c);
    % prepare training samples
    subsetIndex = SubSampling(trainset, sampleRate, nClass);     
    %subsetIndex = (1:NumTrainSet)';
    % learn projections for one code
    learnedProjection(K*(c-1)+1:K*c,:) =CrossEntropyLoss(...
                       normalizedProjection(K*(c-1)+1:K*c,:), trainset,...
                       subsetIndex, dim*numFeature+1, K, learningRate, threshold,...
                       alpha1, lambda1, lambda2, ...
                       MaxEpoch, ...
                       positivePairSampleRate, negativePairSampleRate,...
                       timeDuration);
                   
                   %CrossEntropyLoss
                   % LinearLoss
end
elapse = toc(tstart);

fprintf('Training time elapse = %.2f sec\n', elapse); 
nameLearn = ['..\dataset\HAPT\noBoosting\Learned_earlyfuse_thre_' num2str(threshold) 'L' ...
        num2str(CodeNum) '_K' num2str(K) '_sub' num2str(sub) '.mat']
save(nameLearn, 'learnedProjection', '-v7.3');

clear learnedProjection;
clear trainset;
clear sub;
clear featIndex;

end

                                