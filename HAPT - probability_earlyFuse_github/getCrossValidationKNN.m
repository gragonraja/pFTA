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

% find the best knn for testing by cross-validation on training set
function accuracyAll = getCrossValidationKNN(sub, num, dataset, learnedProjection, ...
    CodeNum, K, threshold, nClass, alpha, maxKnn)
    % subject-based leave-one-out
    subject = 1:30;
    tmpIDStart = (sub-1)* num+1;
    tmpIDEnd = sub*num;
    remainSub = subject(subject<tmpIDStart | subject>tmpIDEnd);
    timeDuration = dataset.nframe(1);
    accuracyAll = [];
    % for fold cross-validation
    for cv = 1:4
        testingSub = remainSub((cv-1)*num+1:cv*num);
        testMin = min(testingSub);
        testMax = max(testingSub);
        
        testingIndex = dataset.subject>=testMin & dataset.subject<=testMax;
        trainingIndex = dataset.subject<testMin | dataset.subject>testMax;
        
        testset.subject = dataset.subject(testingIndex);
        testset.nframe = dataset.nframe(testingIndex);
        testset.class = dataset.class(testingIndex);
       
        firstindex = (find(testingIndex,1)-1)*timeDuration+1;
        lastindex = find(testingIndex,1, 'last')*timeDuration;
        testset.feats = dataset.feats(firstindex:lastindex,:);
        
        trainset.subject = dataset.subject(trainingIndex);
        trainset.nframe = dataset.nframe(trainingIndex);
        trainset.class = dataset.class(trainingIndex); 
        trainset.feats = [dataset.feats(1:firstindex-1,:); dataset.feats(lastindex+1:end,:)];
        
        [SampleHashCodes, TestSampleCodes]= ...
                    FTAHash(learnedProjection, trainset, testset, ...
                          CodeNum, K, threshold, timeDuration, alpha);
        
        HammingDistanceArray = pdist2(TestSampleCodes, SampleHashCodes, 'hamming'); 
        accuracy = computeAccuray(HammingDistanceArray, testset.class,...
                                                trainset.class, maxKnn, nClass); 
        accuracyAll = [accuracyAll accuracy];
    end
   
end