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

function accuracy = computeAccuraySingle(HammingDistanceArray, TestSampleLabels,...
                                                SampleLabels, knn, nClass)
                                                                                                                 
[TestSize, ~] = size(TestSampleLabels);

TP = 0;

[~,index] = sort(HammingDistanceArray,2,'ascend');

for k = 1: TestSize
    label= knnClassifier(SampleLabels(index(k,1:knn)), nClass);
    if label == TestSampleLabels(k);
        TP = TP+1;
    end
end
accuracy = TP/TestSize;                                              

end