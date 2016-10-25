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


% script for batch running
clear all;
seed = 'default'
addpath '../mtimesx';
CodeNum =128;
K=3;
dim = 6;
alpha=5;
knn = [13 14 13 15 10];
threshold=[1.3 1.2 1.3 1.6 1.6];
accuracy = [];
for sub = 1:5
    tmp = lateFusionBatch(seed, CodeNum, K, knn(sub), sub, threshold(sub), dim, alpha); 
    fprintf('sub: %d, max accuracy = %.4f\n', sub, tmp);
    accuracy = [accuracy tmp];
end
meanAvg = mean(accuracy,2);
[maxAvg,index]= max(meanAvg);
fprintf('mean accuracy = %.4f\n', maxAvg);