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
threshold = 1.6
knn = 10
accuracy=[];
sub = 5;
tmp = lateFusionBatch(seed, CodeNum, K, knn, sub, threshold, dim, alpha); 
[maxR, maxI]=max(tmp);
fprintf('sub: %d, max accuracy = %.4f @ %d\n', sub, maxR, maxI);
accuracy = [accuracy tmp];


meanAvg = mean(accuracy,2);
[maxAvg,index]= max(meanAvg);
fprintf('mean accuracy = %.4f @ knn = %d\n', maxAvg, index);